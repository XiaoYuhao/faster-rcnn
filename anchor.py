import torch
import numpy as np 

def init_anchor(img_size=(800,800), sub_sample=16):
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]

    x_size = img_size[1]
    y_size = img_size[0]
    feature_size_x = (x_size // sub_sample)
    feature_size_y = (y_size // sub_sample)

    ctr_x = np.arange(sub_sample, (feature_size_x + 1) * sub_sample, sub_sample)
    ctr_y = np.arange(sub_sample, (feature_size_y + 1) * sub_sample, sub_sample)

    index = 0
    ctr = dict()
    half_sub_sample = sub_sample // 2
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index] = [-1, -1]
            ctr[index][1] = ctr_x[x] - half_sub_sample
            ctr[index][0] = ctr_y[y] - half_sub_sample
            index += 1

    anchors = np.zeros(((feature_size_x * feature_size_y * 9), 4))
    index = 0
    for c in ctr:
        ctr_y, ctr_x = ctr[c]
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
                w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratios[i])
                anchors[index, 0] = ctr_y - h / 2.
                anchors[index, 1] = ctr_x - w / 2.
                anchors[index, 2] = ctr_y + h / 2.
                anchors[index, 3] = ctr_x + w / 2.
                index += 1
    
    valid_anchor_index = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= y_size) &
        (anchors[:, 3] <= x_size)
    )[0]

    vaild_anchor_boxes = anchors[valid_anchor_index]

    return anchors, vaild_anchor_boxes, valid_anchor_index

def compute_iou(vaild_anchor_boxes, bbox):
    vaild_anchor_num = len(vaild_anchor_boxes)
    bbox_num = len(bbox)
    ious = np.empty((vaild_anchor_num, bbox_num), dtype=np.float32)
    ious.fill(0)
    for num1, i in enumerate(vaild_anchor_boxes):
        ya1, xa1, ya2, xa2 = i
        anchor_area = (ya2- ya1) * (xa2 - xa1)
        for num2, j in enumerate(bbox):
            yb1, xb1, yb2, xb2 = j
            box_area = (yb2 - yb1) * (xb2 - xb1)
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                iou = inter_area / (anchor_area + box_area - inter_area)
            else:
                iou = 0.
            
            ious[num1, num2] = iou
        
    return ious

def get_pos_neg_sample(ious, vaild_anchor_len, pos_iou_threshold=0.7, neg_iou_threshold=0.3, pos_ratio=0.5, n_sample=256):
    gt_argmax_ious = ious.argmax(axis=0)
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
    argmax_ious = ious.argmax(axis=1)
    max_ious = ious[np.arange(vaild_anchor_len), argmax_ious]

    gt_argmax_ious = np.where(ious == gt_max_ious)[0]
    label = np.empty((vaild_anchor_len,), dtype=np.int32)
    label.fill(-1)
    label[max_ious < neg_iou_threshold] = 0
    label[gt_argmax_ious] = 1
    label[max_ious >= pos_iou_threshold] = 1

    n_pos = pos_ratio * n_sample
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1
    
    n_neg = n_sample - np.sum(label == 1)
    neg_index = np.where(label == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index ,size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1
    
    return label, argmax_ious

def get_predict_bbox(anchors, pred_anchor_locs, objectness_score, n_train_pre_nms=12000, min_size=16, image_size=(800,800)):
    anc_height = anchors[:, 2] - anchors[:, 0]
    anc_width = anchors[:, 3] - anchors[:, 1]
    anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
    anc_ctr_x = anchors[: ,1] + 0.5 * anc_width

    pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()
    objectness_score_numpy = objectness_score[0].data.numpy()
    dy = pred_anchor_locs_numpy[:, 0::4]
    dx = pred_anchor_locs_numpy[:, 1::4]
    dh = pred_anchor_locs_numpy[:, 2::4]
    dw = pred_anchor_locs_numpy[:, 3::4]
    ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    h = np.exp(dh) * anc_height[:, np.newaxis]
    w = np.exp(dw) * anc_width[:, np.newaxis]

    roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=pred_anchor_locs_numpy.dtype)
    roi[:, 0::4] = ctr_y - 0.5 * h 
    roi[:, 1::4] = ctr_x - 0.5 * w 
    roi[:, 2::4] = ctr_y + 0.5 * h 
    roi[:, 3::4] = ctr_x + 0.5 * w

    roi[:, slice(0,4,2)] = np.clip(roi[:, slice(0,4,2)], 0, image_size[0])  
    roi[:, slice(1,4,2)] = np.clip(roi[:, slice(1,4,2)], 0, image_size[1])

    hs = roi[:, 2] - roi[:, 0]
    ws = roi[:, 3] - roi[:, 1]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    roi = roi[keep, :]
    score = objectness_score_numpy[keep]

    order = score.ravel().argsort()[::-1]           #选出正样本
    order = score[:n_train_pre_nms]

    return roi, score, order

def nms(roi, score, order, nms_thresh=0.7, n_train_post_nms=2000):
    roi = roi[order, :]
    score = score[order]
    y1 = roi[:, 0]
    x1 = roi[:, 1]
    y2 = roi[:, 2]
    x2 = roi[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]
    keep=[]
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    
    keep = keep[:n_train_post_nms]
    roi = roi[keep]

    return roi


def get_propose_target(roi, bbox, labels, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
    ious = compute_iou(roi, bbox)
    gt_assignment = ious.argmax(axis=1)
    max_iou = ious.max(axis=1)

    gt_roi_label = labels[gt_assignment]

    pos_roi_per_image = n_sample * pos_ratio
    pos_index = np.where(max_iou >= pos_iou_thresh)[0]
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
    
    neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
    neg_roi_per_this_image = n_sample - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
    
    keep_index = np.append(pos_index, neg_index)
    gt_roi_labels = gt_roi_label[keep_index]
    gt_roi_labels[pos_roi_per_this_image:] = 0
    sample_roi = roi[keep_index]

    return sample_roi, keep_index, gt_assignment, gt_roi_labels

def get_coefficient(anchor, bbox):
    height = anchor[:, 2] - anchor[:, 0]
    width = anchor[:, 3] - anchor[:, 1]
    ctr_y = anchor[:, 0] + 0.5 * height
    ctr_x = anchor[:, 1] + 0.5 * width
    base_height = bbox[:, 2] - bbox[:, 0]
    base_width = bbox[:, 3] - bbox[:, 1]
    base_ctr_y = bbox[:, 0] + 0.5 * base_height
    base_ctr_x = bbox[:, 1] + 0.5 *base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height) 
    dw = np.log(base_width / width)

    gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()
    gt_roi_locs = gt_roi_locs.astype(np.float32)
    #print(gt_roi_locs.dtype)

    return gt_roi_locs

