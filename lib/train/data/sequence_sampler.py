import random
import torch.utils.data
import numpy as np
from lib.utils import TensorDict


class SequenceSampler(torch.utils.data.Dataset):
    """
    Sample sequence for sequence-level training
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, frame_sample_mode='sequential', max_interval=10, prob=0.7):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the search frames.\
            max_interval - Maximum interval between sampled frames
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the search frames are sampled in a causally,
                                otherwise randomly within the interval.
            prob - sequential sampling by prob / interval sampling by 1-prob
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.max_interval = max_interval
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.frame_sample_mode = frame_sample_mode
        self.prob=prob
        self.extra=1

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)


    def _sequential_sample(self, visible):
        # Sample frames in sequential manner
        template_frame_ids = self._sample_visible_ids(visible, num_ids=1, min_id=0,
                                                   max_id=len(visible) - self.num_search_frames)
        if self.max_gap == -1:
            left = template_frame_ids[0]
        else:
            # template frame (1) ->(max_gap) -> search frame (num_search_frames)
            left_max = min(len(visible) - self.num_search_frames, template_frame_ids[0] + self.max_gap)
            left = self._sample_visible_ids(visible, num_ids=1, min_id=template_frame_ids[0],
                                            max_id=left_max)[0]

        valid_ids = [i for i in range(left, len(visible)) if visible[i]]
        search_frame_ids = valid_ids[:self.num_search_frames]

        # if length is not enough
        last = search_frame_ids[-1]
        while len(search_frame_ids) < self.num_search_frames:
            if last >= len(visible) - 1:
                search_frame_ids.append(last)
            else:
                last += 1
                if visible[last]:
                    search_frame_ids.append(last)

        return template_frame_ids, search_frame_ids


    def _random_interval_sample(self, visible):
        # Get valid ids
        valid_ids = [i for i in range(len(visible)) if visible[i]]

        # Sample template frame
        avg_interval = self.max_interval
        while avg_interval * (self.num_search_frames - 1) > len(visible):
            avg_interval = max(avg_interval - 1, 1)

        while True:
            template_frame_ids = self._sample_visible_ids(visible, num_ids=1, min_id=0,
                                                       max_id=len(visible) - avg_interval * (self.num_search_frames - 1))
            if template_frame_ids == None:
                avg_interval = avg_interval - 1
            else:
                break

            if avg_interval == 0:
                template_frame_ids = [valid_ids[0]]
                break

        # Sample first search frame
        if self.max_gap == -1:
            search_frame_ids = template_frame_ids
        else:
            avg_interval = self.max_interval
            while avg_interval * (self.num_search_frames - 1) > len(visible):
                avg_interval = max(avg_interval - 1, 1)

            while True:
                left_max = min(max(len(visible) - avg_interval * (self.num_search_frames - 1), template_frame_ids[0] + 1),
                               template_frame_ids[0] + self.max_gap)
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, min_id=template_frame_ids[0],
                                                          max_id=left_max)

                if search_frame_ids == None:
                    avg_interval = avg_interval - 1
                else:
                    break

                if avg_interval == -1:
                    search_frame_ids = template_frame_ids
                    break

        # Sample rest of the search frames with random interval
        last = search_frame_ids[0]
        while last <= len(visible) - 1 and len(search_frame_ids) < self.num_search_frames:
            # sample id with interval
            max_id = min(last + self.max_interval + 1, len(visible))
            id = self._sample_visible_ids(visible, num_ids=1, min_id=last,
                                          max_id=max_id)

            if id is None:
                # If not found in current range, find from previous range
                last = last + self.max_interval
            else:
                search_frame_ids.append(id[0])
                last = search_frame_ids[-1]

        # if length is not enough, randomly sample new ids
        if len(search_frame_ids) < self.num_search_frames:
            valid_ids = [x for x in valid_ids if x > search_frame_ids[0] and x not in search_frame_ids]

            if len(valid_ids) > 0:
                new_ids = random.choices(valid_ids, k=min(len(valid_ids),
                                                          self.num_search_frames - len(search_frame_ids)))
                search_frame_ids = search_frame_ids + new_ids
                search_frame_ids = sorted(search_frame_ids, key=int)

        # if length is still not enough, duplicate last frame
        while len(search_frame_ids) < self.num_search_frames:
            search_frame_ids.append(search_frame_ids[-1])

        for i in range(1, self.num_search_frames):
            if search_frame_ids[i] - search_frame_ids[i - 1] > self.max_interval:
                print(search_frame_ids[i] - search_frame_ids[i - 1])

        return template_frame_ids, search_frame_ids


    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        if dataset.get_name() == 'got10k' :
            max_gap = self.max_gap
            max_interval = self.max_interval
        else:
            max_gap = self.max_gap
            max_interval = self.max_interval
            self.max_gap = max_gap * self.extra
            self.max_interval = max_interval * self.extra
            
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        while True:
            try:
                enough_visible_frames = False
                while not enough_visible_frames:
                    # Sample a sequence
                    seq_id = random.randint(0, dataset.get_num_sequences() - 1)

                    # Sample frames
                    seq_info_dict = dataset.get_sequence_info(seq_id)
                    visible = seq_info_dict['visible']

                    enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                            self.num_search_frames + self.num_template_frames) and len(visible) >= (self.num_search_frames + self.num_template_frames)

                    enough_visible_frames = enough_visible_frames or not is_video_dataset

                if is_video_dataset:
                    if self.frame_sample_mode == 'sequential':
                        template_frame_ids, search_frame_ids = self._sequential_sample(visible)

                    elif self.frame_sample_mode == 'random_interval':
                        if random.random() < self.prob:
                            template_frame_ids, search_frame_ids = self._random_interval_sample(visible)
                        else:
                            template_frame_ids, search_frame_ids = self._sequential_sample(visible)
                    else:
                        self.max_gap = max_gap
                        self.max_interval = max_interval
                        raise NotImplementedError
                else:
                    # In case of image dataset, just repeat the image to generate synthetic video
                    template_frame_ids = [1] * self.num_template_frames
                    search_frame_ids = [1] * self.num_search_frames
                #print(dataset.get_name(), search_frame_ids, self.max_gap, self.max_interval)
                self.max_gap = max_gap
                self.max_interval = max_interval
                #print(self.max_gap, self.max_interval)
                template_frames, template_anno, meta_obj_template = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_search = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                template_bbox = [bbox.numpy() for bbox in template_anno['bbox']] # tensor -> numpy array
                search_bbox = [bbox.numpy() for bbox in search_anno['bbox']] # tensor -> numpy array
                # print("====================================================================================")
                # print("dataset index: {}".format(index))
                # print("seq_id: {}".format(seq_id))
                # print('template_frame_ids: {}'.format(template_frame_ids))
                # print('search_frame_ids: {}'.format(search_frame_ids))
                return TensorDict({'template_images': np.array(template_frames).squeeze(),    # 1 template images
                        'template_annos': np.array(template_bbox).squeeze(),
                        'search_images': np.array(search_frames),      # (num_frames) search images
                        'search_annos': np.array(search_bbox),
                        'seq_id': seq_id,
                        'dataset': dataset.get_name(),
                        'search_class': meta_obj_search.get('object_class_name'),
                        'num_frames': len(search_frames)
                        })
            except Exception:
                pass