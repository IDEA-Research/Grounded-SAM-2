from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import math
import numpy as np
import numpy
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
import lib.train.data.bounding_box_utils as bbutils
from lib.utils.merge import merge_template_search
from torch.distributions.categorical import Categorical
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2 - x1) * (y2 - y1)
    target_a = (tx2 - tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def generate_sa_simdr(joints):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = 48
    image_size = [256, 256]
    simdr_split_ratio = 1.5625
    sigma = 6

    target_x1 = np.zeros((num_joints,
                          int(image_size[0] * simdr_split_ratio)),
                         dtype=np.float32)
    target_y1 = np.zeros((num_joints,
                          int(image_size[1] * simdr_split_ratio)),
                         dtype=np.float32)
    target_x2 = np.zeros((num_joints,
                          int(image_size[0] * simdr_split_ratio)),
                         dtype=np.float32)
    target_y2 = np.zeros((num_joints,
                          int(image_size[1] * simdr_split_ratio)),
                         dtype=np.float32)
    zero_4_begin = np.zeros((num_joints, 1), dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        mu_x1 = joints[joint_id][0]
        mu_y1 = joints[joint_id][1]
        mu_x2 = joints[joint_id][2]
        mu_y2 = joints[joint_id][3]

        x1 = np.arange(0, int(image_size[0] * simdr_split_ratio), 1, np.float32)
        y1 = np.arange(0, int(image_size[1] * simdr_split_ratio), 1, np.float32)
        x2 = np.arange(0, int(image_size[0] * simdr_split_ratio), 1, np.float32)
        y2 = np.arange(0, int(image_size[1] * simdr_split_ratio), 1, np.float32)

        target_x1[joint_id] = (np.exp(- ((x1 - mu_x1) ** 2) / (2 * sigma ** 2))) / (
                sigma * np.sqrt(np.pi * 2))
        target_y1[joint_id] = (np.exp(- ((y1 - mu_y1) ** 2) / (2 * sigma ** 2))) / (
                sigma * np.sqrt(np.pi * 2))
        target_x2[joint_id] = (np.exp(- ((x2 - mu_x2) ** 2) / (2 * sigma ** 2))) / (
                sigma * np.sqrt(np.pi * 2))
        target_y2[joint_id] = (np.exp(- ((y2 - mu_y2) ** 2) / (2 * sigma ** 2))) / (
                sigma * np.sqrt(np.pi * 2))
    return target_x1, target_y1, target_x2, target_y2


# angle cost
def SIoU_loss(test1, test2, theta=4):
    eps = 1e-7
    cx_pred = (test1[:, 0] + test1[:, 2]) / 2
    cy_pred = (test1[:, 1] + test1[:, 3]) / 2
    cx_gt = (test2[:, 0] + test2[:, 2]) / 2
    cy_gt = (test2[:, 1] + test2[:, 3]) / 2

    dist = ((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2 * torch.sin(torch.arcsin(x) - torch.pi / 4) ** 2
    # distance cost
    xmin = torch.min(test1[:, 0], test2[:, 0])
    xmax = torch.max(test1[:, 2], test2[:, 2])
    ymin = torch.min(test1[:, 1], test2[:, 1])
    ymax = torch.max(test1[:, 3], test2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw + eps)) ** 2
    py = ((cy_gt - cy_pred) / (ch + eps)) ** 2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    # shape cost
    w_pred = test1[:, 2] - test1[:, 0]
    h_pred = test1[:, 3] - test1[:, 1]
    w_gt = test2[:, 2] - test2[:, 0]
    h_gt = test2[:, 3] - test2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    # IoU loss
    lt = torch.max(test1[..., :2], test2[..., :2])  # [B, rows, 2]
    rb = torch.min(test1[..., 2:], test2[..., 2:])  # [B, rows, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (test1[..., 2] - test1[..., 0]) * (
            test1[..., 3] - test1[..., 1])
    area2 = (test2[..., 2] - test2[..., 0]) * (
            test2[..., 3] - test2[..., 1])
    iou = overlap / (area1 + area2 - overlap)

    SIoU = 1 - iou + (omega + dis) / 2
    return SIoU, iou


def ciou(pred, target, eps=1e-7):
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw ** 2 + ch ** 2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    factor = 4 / math.pi ** 2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v ** 2 / (1 - ious + v))
    return cious, ious


class ARTrackSeqActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, bins, search_size, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bins = bins
        self.search_size = search_size
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.focal = None
        self.range = cfg.MODEL.RANGE
        self.pre_num = cfg.MODEL.PRENUM
        self.loss_weight['KL'] = 0
        self.loss_weight['focal'] = 0
        self.pre_bbox = None
        self.x_feat_rem = None
        self.update_rem = None

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            try:
                im_patch = cv2.resize(im_patch, (model_sz, model_sz))
            except:
                return None
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

    def batch_init(self, images, template_bbox, initial_bbox) -> dict:
        self.frame_num = 1
        self.device = 'cuda'
        # Convert bbox (x1, y1, w, h) -> (cx, cy, w, h)

        template_bbox = bbutils.batch_xywh2center2(template_bbox)  # ndarray:(2*num_seq,4)
        initial_bbox = bbutils.batch_xywh2center2(initial_bbox)  # ndarray:(2*num_seq,4)
        self.center_pos = initial_bbox[:, :2]  # ndarray:(2*num_seq,2)
        self.size = initial_bbox[:, 2:]  # ndarray:(2*num_seq,2)
        self.pre_bbox = initial_bbox
        for i in range(self.pre_num - 1):
            self.pre_bbox = numpy.concatenate((self.pre_bbox, initial_bbox), axis=1)
        # print(self.pre_bbox.shape)

        template_factor = self.cfg.DATA.TEMPLATE.FACTOR
        w_z = template_bbox[:, 2] * template_factor  # ndarray:(2*num_seq)
        h_z = template_bbox[:, 3] * template_factor  # ndarray:(2*num_seq)
        s_z = np.ceil(np.sqrt(w_z * h_z))  # ndarray:(2*num_seq)

        self.channel_average = []
        for img in images:
            self.channel_average.append(np.mean(img, axis=(0, 1)))
        self.channel_average = np.array(self.channel_average)  # ndarray:(2*num_seq,3)

        # get crop
        z_crop_list = []
        for i in range(len(images)):
            here_crop = self.get_subwindow(images[i], template_bbox[i, :2],
                                           self.cfg.DATA.TEMPLATE.SIZE, s_z[i], self.channel_average[i])
            z_crop = here_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.inplace = False
            z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
            z_crop_list.append(z_crop.clone())
        z_crop = torch.cat(z_crop_list, dim=0)  # Tensor(2*num_seq,3,128,128)

        self.update_rem = None

        out = {'template_images': z_crop}
        return out

    def batch_track(self, img, gt_boxes, template, action_mode='max') -> dict:
        search_factor = self.cfg.DATA.SEARCH.FACTOR
        w_x = self.size[:, 0] * search_factor
        h_x = self.size[:, 1] * search_factor
        s_x = np.ceil(np.sqrt(w_x * h_x))

        gt_boxes_corner = bbutils.batch_xywh2corner(gt_boxes)  # ndarray:(2*num_seq,4)

        x_crop_list = []
        gt_in_crop_list = []
        pre_seq_list = []
        pre_seq_in_list = []
        x_feat_list = []

        magic_num = (self.range - 1) * 0.5
        for i in range(len(img)):
            channel_avg = np.mean(img[i], axis=(0, 1))
            x_crop = self.get_subwindow(img[i], self.center_pos[i], self.cfg.DATA.SEARCH.SIZE,
                                        round(s_x[i]), channel_avg)
            if x_crop == None:
                return None
            for q in range(self.pre_num):
                pre_seq_temp = bbutils.batch_center2corner(self.pre_bbox[:, 0 + 4 * q:4 + 4 * q])
                if q == 0:
                    pre_seq = pre_seq_temp
                else:
                    pre_seq = numpy.concatenate((pre_seq, pre_seq_temp), axis=1)

            if gt_boxes_corner is not None and np.sum(np.abs(gt_boxes_corner[i] - np.zeros(4))) > 10:
                pre_in = np.zeros(4 * self.pre_num)
                for w in range(self.pre_num):

                    pre_in[0 + w * 4:2 + w * 4] = pre_seq[i, 0 + w * 4:2 + w * 4] - self.center_pos[i]
                    pre_in[2 + w * 4:4 + w * 4] = pre_seq[i, 2 + w * 4:4 + w * 4] - self.center_pos[i]
                    pre_in[0 + w * 4:4 + w * 4] = pre_in[0 + w * 4:4 + w * 4] * (
                                self.cfg.DATA.SEARCH.SIZE / s_x[i]) + self.cfg.DATA.SEARCH.SIZE / 2
                    pre_in[0 + w * 4:4 + w * 4] = pre_in[0 + w * 4:4 + w * 4] / self.cfg.DATA.SEARCH.SIZE

                pre_seq_list.append(pre_in)
                gt_in_crop = np.zeros(4)
                gt_in_crop[:2] = gt_boxes_corner[i, :2] - self.center_pos[i]
                gt_in_crop[2:] = gt_boxes_corner[i, 2:] - self.center_pos[i]
                gt_in_crop = gt_in_crop * (self.cfg.DATA.SEARCH.SIZE / s_x[i]) + self.cfg.DATA.SEARCH.SIZE / 2
                gt_in_crop[2:] = gt_in_crop[2:] - gt_in_crop[:2]  # (x1,y1,x2,y2) to (x1,y1,w,h)
                gt_in_crop_list.append(gt_in_crop)
            else:
                pre_in = np.zeros(4 * self.pre_num)
                pre_seq_list.append(pre_in)
                gt_in_crop_list.append(np.zeros(4))
            pre_seq_input = torch.from_numpy(pre_in).clamp(-1 * magic_num, 1 + magic_num)
            pre_seq_input = (pre_seq_input + 0.5) * (self.bins - 1)
            pre_seq_in_list.append(pre_seq_input.clone())
            x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
            x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)
            x_crop_list.append(x_crop.clone())

        x_crop = torch.cat(x_crop_list, dim=0)
        pre_seq_output = torch.cat(pre_seq_in_list, dim=0).reshape(-1, 4 * self.pre_num)

        outputs = self.net(template, x_crop, seq_input=pre_seq_output, head_type=None, stage="batch_track",
                           search_feature=self.x_feat_rem, update=None)
        selected_indices = outputs['seqs'].detach()
        x_feat = outputs['x_feat'].detach().cpu()
        self.x_feat_rem = x_feat.clone()
        x_feat_list.append(x_feat.clone())

        pred_bbox = selected_indices[:, 0:4].data.cpu().numpy()
        bbox = (pred_bbox / (self.bins - 1) - magic_num) * s_x.reshape(-1, 1)
        cx = bbox[:, 0] + self.center_pos[:, 0] - s_x / 2
        cy = bbox[:, 1] + self.center_pos[:, 1] - s_x / 2
        width = bbox[:, 2] - bbox[:, 0]
        height = bbox[:, 3] - bbox[:, 1]
        cx = cx + width / 2
        cy = cy + height / 2

        for i in range(len(img)):
            cx[i], cy[i], width[i], height[i] = self._bbox_clip(cx[i], cy[i], width[i],
                                                                height[i], img[i].shape[:2])
        self.center_pos = np.stack([cx, cy], 1)
        self.size = np.stack([width, height], 1)
        for e in range(self.pre_num):
            if e != self.pre_num - 1:
                self.pre_bbox[:, 0 + e * 4:4 + e * 4] = self.pre_bbox[:, 4 + e * 4:8 + e * 4]
            else:
                self.pre_bbox[:, 0 + e * 4:4 + e * 4] = numpy.stack([cx, cy, width, height], 1)

        bbox = np.stack([cx - width / 2, cy - height / 2, width, height], 1)

        out = {
            'search_images': x_crop,
            'pred_bboxes': bbox,
            'selected_indices': selected_indices.cpu(),
            'gt_in_crop': torch.tensor(np.stack(gt_in_crop_list, axis=0), dtype=torch.float),
            'pre_seq': torch.tensor(np.stack(pre_seq_list, axis=0), dtype=torch.float),
            'x_feat': torch.tensor([item.cpu().detach().numpy() for item in x_feat_list], dtype=torch.float),
        }

        return out

    def explore(self, data):
        results = {}
        search_images_list = []
        search_anno_list = []
        iou_list = []
        pre_seq_list = []
        x_feat_list = []

        num_frames = data['num_frames']
        images = data['search_images']
        gt_bbox = data['search_annos']
        template = data['template_images']
        template_bbox = data['template_annos']

        template = template
        template_bbox = template_bbox
        template_bbox = np.array(template_bbox)
        num_seq = len(num_frames)

        for idx in range(np.max(num_frames)):
            here_images = [img[idx] for img in images]  # S, N
            here_gt_bbox = np.array([gt[idx] for gt in gt_bbox])

            here_images = here_images
            here_gt_bbox = np.concatenate([here_gt_bbox], 0)

            if idx == 0:
                outputs_template = self.batch_init(template, template_bbox, here_gt_bbox)
                results['template_images'] = outputs_template['template_images']

            else:
                outputs = self.batch_track(here_images, here_gt_bbox, outputs_template['template_images'],
                                           action_mode='half')
                if outputs == None:
                    return None

                x_feat = outputs['x_feat']
                pred_bbox = outputs['pred_bboxes']
                search_images_list.append(outputs['search_images'])
                search_anno_list.append(outputs['gt_in_crop'])
                if len(outputs['pre_seq']) != 8:
                    print(outputs['pre_seq'])
                    print(len(outputs['pre_seq']))
                    print(idx)
                    print(data['num_frames'])
                    print(data['search_annos'])
                    return None
                pre_seq_list.append(outputs['pre_seq'])
                pred_bbox_corner = bbutils.batch_xywh2corner(pred_bbox)
                gt_bbox_corner = bbutils.batch_xywh2corner(here_gt_bbox)
                here_iou = []
                for i in range(num_seq):
                    bbox_iou = IoU(pred_bbox_corner[i], gt_bbox_corner[i])
                    here_iou.append(bbox_iou)
                iou_list.append(here_iou)
                x_feat_list.append(x_feat.clone())

        results['x_feat'] = torch.cat([torch.stack(x_feat_list)], dim=2)

        results['search_images'] = torch.cat([torch.stack(search_images_list)],
                                             dim=1)
        results['search_anno'] = torch.cat([torch.stack(search_anno_list)],
                                           dim=1)
        results['pre_seq'] = torch.cat([torch.stack(pre_seq_list)], dim=1)

        iou_tensor = torch.tensor(iou_list, dtype=torch.float)
        results['baseline_iou'] = torch.cat([iou_tensor[:, :num_seq]], dim=1)


        return results

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]
        gt_bbox = data['search_anno'][-1]
        begin = self.bins
        end = self.bins + 1
        gt_bbox[:, 2] = gt_bbox[:, 0] + gt_bbox[:, 2]
        gt_bbox[:, 3] = gt_bbox[:, 1] + gt_bbox[:, 3]
        gt_bbox = gt_bbox.clamp(min=0.5, max=1.5)
        data['real_bbox'] = gt_bbox
        seq_ori = gt_bbox * (self.bins - 1)
        seq_ori = seq_ori.int().to(search_img)
        B = seq_ori.shape[0]
        seq_input = torch.cat([torch.ones((B, 1)).to(search_img) * begin, seq_ori], dim=1)
        seq_output = torch.cat([seq_ori, torch.ones((B, 1)).to(search_img) * end], dim=1)
        data['seq_input'] = seq_input
        data['seq_output'] = seq_output
        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            seq_input=seq_input)

        return out_dict

    def compute_sequence_losses(self, data):
        num_frames = data['search_images'].shape[0]
        template_images = data['template_images'].repeat(num_frames, 1, 1, 1, 1)
        template_images = template_images.view(-1, *template_images.size()[2:])
        search_images = data['search_images'].reshape(-1, *data['search_images'].size()[2:])
        search_anno = data['search_anno'].reshape(-1, *data['search_anno'].size()[2:])

        magic_num = (self.range - 1) * 0.5
        self.loss_weight['focal'] = 0
        pre_seq = data['pre_seq'].reshape(-1, 4 * self.pre_num)
        x_feat = data['x_feat'].reshape(-1, *data['x_feat'].size()[2:])
        pre_seq = pre_seq.clamp(-1 * magic_num, 1 + magic_num)
        pre_seq = (pre_seq + magic_num) * (self.bins - 1)

        outputs = self.net(template_images, search_images, seq_input=pre_seq, stage="forward_pass",
                           search_feature=x_feat, update=None)

        pred_feat = outputs["feat"]
        # generate labels
        if self.focal == None:
            weight = torch.ones(self.bins * self.range + 2) * 1
            weight[self.bins * self.range + 1] = 0.1
            weight[self.bins * self.range] = 0.1
            weight.to(pred_feat)
            self.focal = torch.nn.CrossEntropyLoss(weight=weight, size_average=True).to(pred_feat)

        search_anno[:, 2] = search_anno[:, 2] + search_anno[:, 0]
        search_anno[:, 3] = search_anno[:, 3] + search_anno[:, 1]
        target = (search_anno / self.cfg.DATA.SEARCH.SIZE + 0.5) * (self.bins - 1)

        target = target.clamp(min=0.0, max=(self.bins * self.range - 0.0001))
        target_iou = target
        target = torch.cat([target], dim=1)
        target = target.reshape(-1).to(torch.int64)
        pred = pred_feat.permute(1, 0, 2).reshape(-1, self.bins * self.range + 2)
        varifocal_loss = self.focal(pred, target)
        pred = pred_feat[0:4, :, 0:self.bins * self.range]
        target = target_iou[:, 0:4].to(pred_feat) / (self.bins - 1) - magic_num
        out = pred.softmax(-1).to(pred)
        mul = torch.range(-1 * magic_num + 1 / (self.bins * self.range), 1 + magic_num - 1 / (self.bins * self.range), 2 / (self.bins * self.range)).to(pred)
        ans = out * mul
        ans = ans.sum(dim=-1)
        ans = ans.permute(1, 0).to(pred)
        extra_seq = ans
        extra_seq = extra_seq.to(pred)

        cious, iou = SIoU_loss(extra_seq, target, 4)
        cious = cious.mean()

        giou_loss = cious
        loss_bb = self.loss_weight['giou'] * giou_loss + self.loss_weight[
            'focal'] * varifocal_loss

        total_losses = loss_bb

        mean_iou = iou.detach().mean()
        status = {"Loss/total": total_losses.item(),
                  "Loss/giou": giou_loss.item(),
                  "Loss/location": varifocal_loss.item(),
                  "IoU": mean_iou.item()}

        return total_losses, status

