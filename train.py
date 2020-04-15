import torch
import torchvision
from PIL import Image
import numpy as np 
from model import VGG
import anchor
import utils
import transforms

def model(image, targets):
    bbox = np.asarray(targets['boxes'], dtype=np.float32)
    labels = np.asarray(targets['labels'], dtype=np.int8)

    img_var = torch.autograd.Variable(torch.unsqueeze(image, dim=0))
    print(img_var.shape)

    anchors, vaild_anchor_boxes, valid_anchor_index = anchor.init_anchor(img_size=(image.size()[1],image.size()[2]))
    ious = anchor.compute_iou(vaild_anchor_boxes, bbox)
    vaild_anchor_len = len(vaild_anchor_boxes)
    label, argmax_ious = anchor.get_pos_neg_sample(ious, vaild_anchor_len, pos_iou_threshold=0.7, neg_iou_threshold=0.3, pos_ratio=0.5, n_sample=256)

    print(anchors.shape)
    print(vaild_anchor_boxes.shape)
    
    print(np.sum(label == 1))
    print(np.sum(label == 0))

    max_iou_bbox = bbox[argmax_ious]
    anchor_locs = anchor.get_coefficient(vaild_anchor_boxes, max_iou_bbox)  #所有anchor的平移缩放系数
    anchor_cls = np.empty((len(anchors),), dtype=label.dtype)
    anchor_cls.fill(-1)
    anchor_cls[valid_anchor_index] = label

    anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
    anchor_locations.fill(0)
    anchor_locations[valid_anchor_index,:] = anchor_locs                    #有效anchor的平移缩放系数

    vgg = VGG()
    out_map, pred_anchor_locs, pred_anchor_cls = vgg.forward(img_var)
    print(out_map.data.shape)
    print(out_map.shape)
    print(pred_anchor_locs.shape)
    print(pred_anchor_cls.shape)

    pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
    print(pred_anchor_locs.shape)

    pred_anchor_cls = pred_anchor_cls.permute(0, 2, 3, 1).contiguous()
    print(pred_anchor_cls.shape)
    objectness_score = pred_anchor_cls.view(1, 67, 120, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
    print(objectness_score.shape)

    pred_anchor_cls = pred_anchor_cls.view(1, -1, 2)
    print(pred_anchor_cls.shape)

    # 计算RPN网络的损失
    rpn_anchor_loc = pred_anchor_locs[0]
    rpn_anchor_cls = pred_anchor_cls[0]
    anchor_locations = torch.from_numpy(anchor_locations)
    anchor_cls = torch.from_numpy(anchor_cls)
    print(rpn_anchor_loc.shape, rpn_anchor_cls.shape, anchor_locations.shape, anchor_cls.shape)

    print(rpn_anchor_loc.dtype, rpn_anchor_cls.dtype, anchor_locations.dtype, anchor_cls.dtype)
    rpn_loss = vgg.rpn_loss(rpn_anchor_loc, rpn_anchor_cls, anchor_locations, anchor_cls, rpn_lambda=10.0)
    print("rpn_loss: {}".format(rpn_loss))


    # 根据rpn预测出的anchor平移放缩系数以及anchor坐标的出预测框的坐标
    roi, score, order = anchor.get_predict_bbox(anchor. pred_anchor_locs, objectness_score, n_train_pre_nms=12000, min_size=16)
    roi = anchor.nms(roi, score, order, nms_thresh=0.7, n_train_post_nms=2000)

    #sample_roi, keep_index, gt_assignment, roi_labels = anchor.get_propose_target


labels_dict = {
            'background' : 0,
            'ore carrier' : 1,
            'passenger ship' : 2,
            'container ship' : 3,
            'bulk cargo carrier' : 4,
            'general cargo ship' : 5,
            'fishing boat' : 6
        }
classes_name =['background','ore carrier','passenger ship','container ship','bulk cargo carrier','general cargo ship','fishing boat']

if __name__ == '__main__' :
    path = "../SeaShips/JPEGImages/000001.jpg"
    img = Image.open(path)
    anno_path = "../SeaShips/Annotations/000001.xml"
    labels, boxes = utils.readxml(anno_path, labels_dict)
    targets = dict()
    targets['labels'] = labels
    targets['boxes'] = boxes
    func = transforms.ToTensor()
    img, targets = func(img, targets)
    print(img.size())
    print(targets)

    model(img, targets)