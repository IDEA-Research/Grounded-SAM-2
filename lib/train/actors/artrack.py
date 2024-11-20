from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import math
import numpy as np
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

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

    dist = ((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2*torch.sin(torch.arcsin(x)-torch.pi/4)**2
    # distance cost
    xmin = torch.min(test1[:, 0], test2[:, 0])
    xmax = torch.max(test1[:, 2], test2[:, 2])
    ymin = torch.min(test1[:, 1], test2[:, 1])
    ymax = torch.max(test1[:, 3], test2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw+eps))**2
    py = ((cy_gt - cy_pred) / (ch+eps))**2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    #shape cost
    w_pred = test1[:, 2] - test1[:, 0]
    h_pred = test1[:, 3] - test1[:, 1]
    w_gt = test2[:, 2] - test2[:, 0]
    h_gt = test2[:, 3] - test2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    #IoU loss
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

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
    return cious, ious

class ARTrackActor(BaseActor):
    """ Actor for training ARTrack models """

    def __init__(self, net, objective, loss_weight, settings, bins, search_size, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bins = bins
        self.range = self.cfg.MODEL.RANGE
        self.search_size = search_size
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.focal = None
        self.loss_weight['KL'] = 100
        self.loss_weight['focal'] = 2

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

        if len(template_list) == 1:
            template_list = template_list[0]
        gt_bbox = data['search_anno'][-1]
        begin = self.bins * self.range
        end = self.bins * self.range + 1

        magic_num = (self.range - 1) * 0.5
        gt_bbox[:, 2] = gt_bbox[:, 0] + gt_bbox[:, 2]
        gt_bbox[:, 3] = gt_bbox[:, 1] + gt_bbox[:, 3]
        gt_bbox = gt_bbox.clamp(min=(-1*magic_num), max=(1+magic_num))
        data['real_bbox'] = gt_bbox

        seq_ori = (gt_bbox + magic_num) * (self.bins - 1)

        seq_ori = seq_ori.int().to(search_img)
        B = seq_ori.shape[0]
        seq_input = torch.cat([torch.ones((B, 1)).to(search_img) * begin, seq_ori], dim=1)
        seq_output = torch.cat([seq_ori, torch.ones((B, 1)).to(search_img) * end], dim=1)
        data['seq_input'] = seq_input
        data['seq_output'] = seq_output
        out_dict = self.net(template=template_list,
                            search=search_img,
                            seq_input=seq_input)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        bins = self.bins
        magic_num = (self.range - 1) * 0.5
        seq_output = gt_dict['seq_output']
        pred_feat = pred_dict["feat"]
        if self.focal == None:
            weight = torch.ones(bins*self.range+2) * 1
            weight[bins*self.range+1] = 0.1
            weight[bins*self.range] = 0.1
            weight.to(pred_feat)
            self.klloss = torch.nn.KLDivLoss(reduction='none').to(pred_feat)

            self.focal = torch.nn.CrossEntropyLoss(weight=weight, size_average=True).to(pred_feat)
        # compute varfifocal loss
        pred = pred_feat.permute(1, 0, 2).reshape(-1, bins*2+2)
        target = seq_output.reshape(-1).to(torch.int64)
        varifocal_loss = self.focal(pred, target)
        # compute giou and L1 loss
        beta = 1
        pred = pred_feat[0:4, :, 0:bins*self.range] * beta
        target = seq_output[:, 0:4].to(pred_feat)
        
        out = pred.softmax(-1).to(pred)
        mul = torch.range((-1*magic_num+1/(self.bins*self.range)), (1+magic_num-1/(self.bins*self.range)), 2/(self.bins*self.range)).to(pred)
        ans = out * mul
        ans = ans.sum(dim=-1)
        ans = ans.permute(1, 0).to(pred)
        target = target / (bins - 1) - magic_num
        extra_seq = ans
        extra_seq = extra_seq.to(pred)
        sious, iou = SIoU_loss(extra_seq, target, 4)
        sious = sious.mean()
        siou_loss = sious
        l1_loss = self.objective['l1'](extra_seq, target)

        loss = self.loss_weight['giou'] * siou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * varifocal_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": siou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": varifocal_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
