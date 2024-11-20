import os
import os.path as osp
import sys
import numpy as np
from lib.test.utils.load_text import load_text
import torch
import pickle
from tqdm import tqdm

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.environment import env_settings


def calc_err_center(pred_bb, anno_bb, normalized=False):
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
    return err_center


def calc_iou_overlap(pred_bb, anno_bb):
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union


def calc_seq_err_robust(pred_bb, anno_bb, dataset, target_visible=None):
    pred_bb = pred_bb.clone()

    # Check if invalid values are present
    if torch.isnan(pred_bb).any() or (pred_bb[:, 2:] < 0.0).any():
        raise Exception('Error: Invalid results')

    if torch.isnan(anno_bb).any():
        if dataset == 'uav':
            pass
        else:
            raise Exception('Warning: NaNs in annotation')

    if (pred_bb[:, 2:] == 0.0).any():
        for i in range(1, pred_bb.shape[0]):
            if i >= anno_bb.shape[0]:
                continue
            if (pred_bb[i, 2:] == 0.0).any() and not torch.isnan(anno_bb[i, :]).any():
                pred_bb[i, :] = pred_bb[i-1, :]

    if pred_bb.shape[0] != anno_bb.shape[0]:
        if dataset == 'lasot':
            if pred_bb.shape[0] > anno_bb.shape[0]:
                # For monkey-17, there is a mismatch for some trackers.
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                raise Exception('Mis-match in tracker prediction and GT lengths')
        else:
            # print('Warning: Mis-match in tracker prediction and GT lengths')
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                pad = torch.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4)).type_as(pred_bb)
                pred_bb = torch.cat((pred_bb, pad), dim=0)

    pred_bb[0, :] = anno_bb[0, :]

    if target_visible is not None:
        target_visible = target_visible.bool()
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    # handle invalid anno cases
    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot':
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid


def extract_results(trackers, dataset, report_name, skip_missing_seq=False, plot_bin_gap=0.05,
                    exclude_invalid_frames=False):
    settings = env_settings()
    eps = 1e-16

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    avg_overlap_all = torch.zeros((len(dataset), len(trackers)), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((len(dataset), len(trackers), threshold_set_overlap.numel()),
                                                dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                               dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                                    dtype=torch.float32)

    from collections import defaultdict
    # default dict of default dict of list
    

    valid_sequence = torch.ones(len(dataset), dtype=torch.uint8)

    for seq_id, seq in enumerate(tqdm(dataset)):
        frame_success_rate_plot_overlap = defaultdict(lambda: defaultdict(list))    
        frame_success_rate_plot_center = defaultdict(lambda: defaultdict(list))
        frame_success_rate_plot_center_norm = defaultdict(lambda: defaultdict(list))
        # Load anno
        anno_bb = torch.tensor(seq.ground_truth_rect)
        target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
        for trk_id, trk in enumerate(trackers):
            # Load results
            base_results_path = '{}/{}'.format(trk.results_dir, seq.name)
            results_path = '{}.txt'.format(base_results_path)

            if os.path.isfile(results_path):
                pred_bb = torch.tensor(load_text(str(results_path), delimiter=('\t', ','), dtype=np.float64))
            else:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    raise Exception('Result not found. {}'.format(results_path))

            # Calculate measures
            err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(
                pred_bb, anno_bb, seq.dataset, target_visible)

            avg_overlap_all[seq_id, trk_id] = err_overlap[valid_frame].mean()

            if exclude_invalid_frames:
                seq_length = valid_frame.long().sum()
            else:
                seq_length = anno_bb.shape[0]

            if seq_length <= 0:
                raise Exception('Seq length zero')

            ave_success_rate_plot_overlap[seq_id, trk_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center[seq_id, trk_id, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center_norm[seq_id, trk_id, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length

            # for frame_id in range(seq_length):
            #     frame_success_rate_plot_overlap[trk_id][frame_id].append((err_overlap[frame_id]).item())
            #     frame_success_rate_plot_center[trk_id][frame_id].append((err_center[frame_id]).item())
            #     frame_success_rate_plot_center_norm[trk_id][frame_id].append((err_center_normalized[frame_id] < 0.2).item())

        # output_folder = "../cvpr2025/per_frame_success_rate"
        # os.makedirs(output_folder, exist_ok=True)
        # with open(osp.join(output_folder, f"{seq.name}.txt"), 'w') as f:
        #     for frame_id in range(seq_length):
        #         suc_score = frame_success_rate_plot_overlap[trk_id][frame_id][0]
        #         f.write(f"{suc_score}\n")

    # # plot the average success rate, center normalized for each tracker 
    # # y axis: success rate
    # # x axis: frame number
    # # different color for each tracker
    # # save the plot as a figure
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # for trk_id, trk in enumerate(trackers):
    #     list_to_plot = [np.mean(frame_success_rate_plot_overlap[trk_id][frame_id]) for frame_id in range(2000)]
    #     # smooth the curve; window size = 10
    #     smooth_list_to_plot = np.convolve(list_to_plot, np.ones((10,))/10, mode='valid')
    #     # the smooth curve and non smooth curve have the same label
    #     plt.plot(smooth_list_to_plot, label=trk.display_name, alpha=1)
    # plt.xlabel('Frame Number')
    # plt.ylabel('Success Rate')
    # plt.title('Average Success Rate Over Frames')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('average_success_rate_plot_overlap.png')
    # plt.close()


    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    # Prepare dictionary for saving data
    seq_names = [s.name for s in dataset]
    tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                     for t in trackers]

    eval_data = {'sequences': seq_names, 'trackers': tracker_names,
                 'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data
