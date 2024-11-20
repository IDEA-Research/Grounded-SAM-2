import os
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
from collections import OrderedDict
from lib.train.admin import env_settings
from lib.utils.lmdb_utils import decode_img, decode_json


def get_target_to_image_ratio(seq):
    anno = torch.Tensor(seq['anno'])
    img_sz = torch.Tensor(seq['image_size'])
    return (anno[0, 2:4].prod() / (img_sz.prod())).sqrt()


class ImagenetVID_lmdb(BaseVideoDataset):
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
        super().__init__("imagenetvid_lmdb", root, image_loader)

        sequence_list_dict = decode_json(root, "cache.json")
        self.sequence_list = sequence_list_dict

        # Filter the sequences based on min_length and max_target_area in the first frame
        self.sequence_list = [x for x in self.sequence_list if len(x['anno']) >= min_length and
                              get_target_to_image_ratio(x) < max_target_area]

    def get_name(self):
        return 'imagenetvid_lmdb'

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
        frame_path = os.path.join('Data', 'VID', 'train', set_name, vid_name,
                                  '{:06d}.JPEG'.format(frame_number))
        return decode_img(self.root, frame_path)

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

