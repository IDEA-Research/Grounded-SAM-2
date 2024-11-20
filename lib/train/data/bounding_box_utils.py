import torch
import numpy as np

def batch_center2corner(boxes):
    xmin = boxes[:, 0] - boxes[:, 2] * 0.5
    ymin = boxes[:, 1] - boxes[:, 3] * 0.5
    xmax = boxes[:, 0] + boxes[:, 2] * 0.5
    ymax = boxes[:, 1] + boxes[:, 3] * 0.5

    if isinstance(boxes, np.ndarray):
        return np.stack([xmin, ymin, xmax, ymax], 1)
    else:
        return torch.stack([xmin, ymin, xmax, ymax], 1)

def batch_corner2center(boxes):
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])

    if isinstance(boxes, np.ndarray):
        return np.stack([cx, cy, w, h], 1)
    else:
        return torch.stack([cx, cy, w, h], 1)

def batch_xywh2center(boxes):
    cx = boxes[:, 0] + (boxes[:, 2] - 1) / 2
    cy = boxes[:, 1] + (boxes[:, 3] - 1) / 2
    w = boxes[:, 2]
    h = boxes[:, 3]

    if isinstance(boxes, np.ndarray):
        return np.stack([cx, cy, w, h], 1)
    else:
        return torch.stack([cx, cy, w, h], 1)

def batch_xywh2center2(boxes):
    cx = boxes[:, 0] + boxes[:, 2] / 2
    cy = boxes[:, 1] + boxes[:, 3] / 2
    w = boxes[:, 2]
    h = boxes[:, 3]

    if isinstance(boxes, np.ndarray):
        return np.stack([cx, cy, w, h], 1)
    else:
        return torch.stack([cx, cy, w, h], 1)


def batch_xywh2corner(boxes):
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 0] + boxes[:, 2]
    ymax = boxes[:, 1] + boxes[:, 3]

    if isinstance(boxes, np.ndarray):
        return np.stack([xmin, ymin, xmax, ymax], 1)
    else:
        return torch.stack([xmin, ymin, xmax, ymax], 1)

def rect_to_rel(bb, sz_norm=None):
    """Convert standard rectangular parametrization of the bounding box [x, y, w, h]
    to relative parametrization [cx/sw, cy/sh, log(w), log(h)], where [cx, cy] is the center coordinate.
    args:
        bb  -  N x 4 tensor of boxes.
        sz_norm  -  [N] x 2 tensor of value of [sw, sh] (optional). sw=w and sh=h if not given.
    """

    c = bb[...,:2] + 0.5 * bb[...,2:]
    if sz_norm is None:
        c_rel = c / bb[...,2:]
    else:
        c_rel = c / sz_norm
    sz_rel = torch.log(bb[...,2:])
    return torch.cat((c_rel, sz_rel), dim=-1)


def rel_to_rect(bb, sz_norm=None):
    """Inverts the effect of rect_to_rel. See above."""

    sz = torch.exp(bb[...,2:])
    if sz_norm is None:
        c = bb[...,:2] * sz
    else:
        c = bb[...,:2] * sz_norm
    tl = c - 0.5 * sz
    return torch.cat((tl, sz), dim=-1)


def masks_to_bboxes(mask, fmt='c'):

    """ Convert a mask tensor to one or more bounding boxes.
    Note: This function is a bit new, make sure it does what it says.  /Andreas
    :param mask: Tensor of masks, shape = (..., H, W)
    :param fmt: bbox layout. 'c' => "center + size" or (x_center, y_center, width, height)
                             't' => "top left + size" or (x_left, y_top, width, height)
                             'v' => "vertices" or (x_left, y_top, x_right, y_bottom)
    :return: tensor containing a batch of bounding boxes, shape = (..., 4)
    """
    batch_shape = mask.shape[:-2]
    mask = mask.reshape((-1, *mask.shape[-2:]))
    bboxes = []

    for m in mask:
        mx = m.sum(dim=-2).nonzero()
        my = m.sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        bboxes.append(bb)

    bboxes = torch.tensor(bboxes, dtype=torch.float32, device=mask.device)
    bboxes = bboxes.reshape(batch_shape + (4,))

    if fmt == 'v':
        return bboxes

    x1 = bboxes[..., :2]
    s = bboxes[..., 2:] - x1 + 1

    if fmt == 'c':
        return torch.cat((x1 + 0.5 * s, s), dim=-1)
    elif fmt == 't':
        return torch.cat((x1, s), dim=-1)

    raise ValueError("Undefined bounding box layout '%s'" % fmt)


def masks_to_bboxes_multi(mask, ids, fmt='c'):
    assert mask.dim() == 2
    bboxes = []

    for id in ids:
        mx = (mask == id).sum(dim=-2).nonzero()
        my = (mask == id).float().sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]

        bb = torch.tensor(bb, dtype=torch.float32, device=mask.device)

        x1 = bb[:2]
        s = bb[2:] - x1 + 1

        if fmt == 'v':
            pass
        elif fmt == 'c':
            bb = torch.cat((x1 + 0.5 * s, s), dim=-1)
        elif fmt == 't':
            bb = torch.cat((x1, s), dim=-1)
        else:
            raise ValueError("Undefined bounding box layout '%s'" % fmt)
        bboxes.append(bb)

    return bboxes
