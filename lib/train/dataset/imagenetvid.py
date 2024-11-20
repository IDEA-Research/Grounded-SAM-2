import os
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import xml.etree.ElementTree as ET
import json
import torch
from collections import OrderedDict
from lib.train.admin import env_settings


def get_target_to_image_ratio(seq):
    anno = torch.Tensor(seq['anno'])
    img_sz = torch.Tensor(seq['image_size'])
    return (anno[0, 2:4].prod() / (img_sz.prod())).sqrt()


class ImagenetVID(BaseVideoDataset):
    """ Imagenet VID dataset.

    Publication:
        ImageNet Large Scale Visual Recognition Challenge
        Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
        Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei
        IJCV, 2015
        https://arxiv.org/pdf/1409.0575.pdf

    Download the dataset from http://image-net.org/
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader, min_length=0, max_target_area=1):
        """
        args:
            root - path to the imagenet vid dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            min_length - Minimum allowed sequence length.
            max_target_area - max allowed ratio between target area and image area. Can be used to filter out targets
                                which cover complete image.
        """
        root = env_settings().imagenet_dir if root is None else root
        super().__init__("imagenetvid", root, image_loader)

        cache_file = os.path.join(root, 'cache.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                sequence_list_dict = json.load(f)

            self.sequence_list = sequence_list_dict
        else:
            # Else process the imagenet annotations and generate the cache file
            self.sequence_list = self._process_anno(root)

            with open(cache_file, 'w') as f:
                json.dump(self.sequence_list, f)

        # Filter the sequences based on min_length and max_target_area in the first frame
        self.sequence_list = [x for x in self.sequence_list if len(x['anno']) >= min_length and
                              get_target_to_image_ratio(x) < max_target_area]

    def get_name(self):
        return 'imagenetvid'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        bb_anno = torch.Tensor(self.sequence_list[seq_id]['anno'])
        valid = (bb_anno[:, 2] > 0) & (bb_anno[:, 3] > 0)
        visible = torch.ByteTensor(self.sequence_list[seq_id]['target_visible']) & valid.byte()
        return {'bbox': bb_anno, 'valid': valid, 'visible': visible}

    def _get_frame(self, sequence, frame_id):
        set_name = 'ILSVRC2015_VID_train_{:04d}'.format(sequence['set_id'])
        vid_name = 'ILSVRC2015_train_{:08d}'.format(sequence['vid_id'])
        frame_number = frame_id + sequence['start_frame']
        frame_path = os.path.join(self.root, 'Data', 'VID', 'train', set_name, vid_name,
                                  '{:06d}.JPEG'.format(frame_number))
        return self.image_loader(frame_path)

    def get_frames(self, seq_id, frame_ids, anno=None):
        sequence = self.sequence_list[seq_id]

        frame_list = [self._get_frame(sequence, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # added the class info to the meta info
        object_meta = OrderedDict({'object_class': sequence['class_name'],
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def _process_anno(self, root):
        # Builds individual tracklets
        base_vid_anno_path = os.path.join(root, 'Annotations', 'VID', 'train')

        all_sequences = []
        for set in sorted(os.listdir(base_vid_anno_path)):
            set_id = int(set.split('_')[-1])
            for vid in sorted(os.listdir(os.path.join(base_vid_anno_path, set))):

                vid_id = int(vid.split('_')[-1])
                anno_files = sorted(os.listdir(os.path.join(base_vid_anno_path, set, vid)))

                frame1_anno = ET.parse(os.path.join(base_vid_anno_path, set, vid, anno_files[0]))
                image_size = [int(frame1_anno.find('size/width').text), int(frame1_anno.find('size/height').text)]

                objects = [ET.ElementTree(file=os.path.join(base_vid_anno_path, set, vid, f)).findall('object')
                           for f in anno_files]

                tracklets = {}

                # Find all tracklets along with start frame
                for f_id, all_targets in enumerate(objects):
                    for target in all_targets:
                        tracklet_id = target.find('trackid').text
                        if tracklet_id not in tracklets:
                            tracklets[tracklet_id] = f_id

                for tracklet_id, tracklet_start in tracklets.items():
                    tracklet_anno = []
                    target_visible = []
                    class_name_id = None

                    for f_id in range(tracklet_start, len(objects)):
                        found = False
                        for target in objects[f_id]:
                            if target.find('trackid').text == tracklet_id:
                                if not class_name_id:
                                    class_name_id = target.find('name').text
                                x1 = int(target.find('bndbox/xmin').text)
                                y1 = int(target.find('bndbox/ymin').text)
                                x2 = int(target.find('bndbox/xmax').text)
                                y2 = int(target.find('bndbox/ymax').text)

                                tracklet_anno.append([x1, y1, x2 - x1, y2 - y1])
                                target_visible.append(target.find('occluded').text == '0')

                                found = True
                                break
                        if not found:
                            break

                    new_sequence = {'set_id': set_id, 'vid_id': vid_id, 'class_name': class_name_id,
                                    'start_frame': tracklet_start, 'anno': tracklet_anno,
                                    'target_visible': target_visible, 'image_size': image_size}
                    all_sequences.append(new_sequence)

        return all_sequences
