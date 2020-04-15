import torch
import numpy as np
import torch.nn as nn

image = torch.zeros((1,3,800,800)).float()

bbox = np.asarray([[20,30,400,500],[300,400,500,600]], dtype=np.float32)
labels = np.asarray([6,8], dtype=np.int8)
#print(bbox)
sub_sample = 16

import torchvision

dummy_img = torch.zeros((1,3,800,800)).float()
#print(dummy_img)

model = torchvision.models.vgg16(pretrained=True)
fe = list(model.features)
#print(fe)

req_features = []
k = dummy_img.clone()
for i in fe:
    k = i(k)
    if k.size()[2]< 800//16:
        break
    #print(i)
    req_features.append(i)
    out_channels = k.size()[1]
#print(len(req_features))
#print(out_channels)

faster_rcnn_fe_extractor = nn.Sequential(*req_features)
out_map = faster_rcnn_fe_extractor(image)
#print(out_map.size())

anchor_ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]
sub_sample = 16

#print(anchor_base)
fe_size = (800//16)
ctr_x = np.arange(16, (fe_size+1)*16, 16)
ctr_y = np.arange(16, (fe_size+1)*16, 16)

#print(ctr_y, ctr_x)
index = 0
ctr = np.zeros((len(ctr_x)*len(ctr_y), 2), dtype=np.int32)
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index, 1] = ctr_x[x] - 8
        ctr[index, 0] = ctr_y[y] - 8
        index += 1

#print(ctr)

index = 0
anchor_base = np.zeros((fe_size*fe_size*len(anchor_ratios)*len(anchor_scales), 4), dtype=np.float32)
for c in ctr:
    ctr_y, ctr_x = c
    for i in range(len(anchor_ratios)):
        for j in range(len(anchor_scales)):
            h = sub_sample * anchor_scales[j] * np.sqrt(anchor_ratios[i])
            w = sub_sample * anchor_scales[j] * np.sqrt(1./anchor_ratios[i])
            anchor_base[index, 0] = ctr_y - h / 2   #y1
            anchor_base[index, 1] = ctr_x - w / 2   #x1
            anchor_base[index, 2] = ctr_y + h / 2   #y2
            anchor_base[index, 3] = ctr_x + w / 2   #x2
            index += 1

#print(anchor_base.shape)

index_inside = np.where(
        (anchor_base[:, 0] >= 0) &
        (anchor_base[:, 1] >= 0) &
        (anchor_base[:, 2] <= 800) &
        (anchor_base[:, 3] <= 800)
    )[0]
#print(index_inside.shape)

label = np.empty((len(index_inside), ), dtype=np.int32)
label.fill(-1)
#print(label.shape)

vaild_anchor_boxes = anchor_base[index_inside]
#print(vaild_anchor_boxes.shape)

ious = np.empty((len(vaild_anchor_boxes), 2), dtype=np.float32)
ious.fill(0)


for num1, i in enumerate(vaild_anchor_boxes):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])

        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1) * (inter_x2- inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0
        
        ious[num1, num2] = iou
    
#print(ious.shape)

#case1
gt_argmax_ious = ious.argmax(axis=0)
#print(gt_argmax_ious)   #gt_argmax_ious数组表示与gt boxes有最大ious值的anchor下标
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
#print(gt_max_ious)      #gt_max_ious数组表示anchors与每个gt boxes的ious最大值

#case2
argmax_ious = ious.argmax(axis=1)
#print(argmax_ious.shape)
#print(argmax_ious)      #argmax_ious数组表示与anchors有最大ious值的gt boxes下标
max_ious= ious[np.arange(len(index_inside)),argmax_ious]
#print(max_ious)         #max_ious数组表示gt boxes与每个anchors的最大ious值

gt_argmax_ious = np.where(ious == gt_max_ious)[0]       #可能会有多个anchors具有相同的ious值
#print(gt_argmax_ious)

pos_iou_threshold = 0.7
neg_iou_threshold = 0.3

label[max_ious < neg_iou_threshold] = 0    #小于阈值的设置为负标签
label[gt_argmax_ious] = 1                  #和gt boxes有最大ious值的设置为正标签
label[max_ious >= pos_iou_threshold] = 1   #大于阈值的设置为正标签

#print(label)

pos_ratio = 0.5
n_sample = 256

n_pos = pos_ratio * n_sample

#positive samples
pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace = False)
    label[disable_index] = -1

#negative samples
n_neg = n_sample - np.sum(label == 1)
neg_index = np.where(label == 0)[0]
if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace = False)
    label[disable_index] = -1

max_iou_bbox = bbox[argmax_ious]
#print(max_iou_bbox)

height = vaild_anchor_boxes[:,2] - vaild_anchor_boxes[:,0]
width = vaild_anchor_boxes[:,3] - vaild_anchor_boxes[:,1]
ctr_y = vaild_anchor_boxes[:,0] + 0.5 * height
ctr_x = vaild_anchor_boxes[:,1] + 0.5 * width

base_height = max_iou_bbox[:,2] - max_iou_bbox[:,0]
base_width = max_iou_bbox[:,3] - max_iou_bbox[:,1]
base_ctr_y = max_iou_bbox[:,0] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:,1] + 0.5 * base_width

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
#print(anchor_locs)

anchor_labels = np.empty((len(anchor_base),), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[index_inside] = label

anchor_locations = np.empty((len(anchor_base), )+ anchor_base.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[index_inside, :] = anchor_locs

#print(anchor_locations.shape)

mid_channels = 512
in_channels = 512
n_anchor = 9
conv1 = nn.Conv2d(in_channels,mid_channels,3,1,1)
reg_layer = nn.Conv2d(mid_channels,n_anchor*4,1,1,0)
cls_layer = nn.Conv2d(mid_channels,n_anchor*2,1,1,0)

conv1.weight.data.normal_(0,0.01)
conv1.bias.data.zero_()

reg_layer.weight.data.normal_(0,0.01)
reg_layer.bias.data.zero_()

cls_layer.weight.data.normal_(0,0.01)
cls_layer.bias.data.zero_()

x = conv1(out_map)
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

#print(pred_anchor_locs.shape,pred_cls_scores.shape)

pred_anchor_locs = pred_anchor_locs.permute(0,2,3,1).contiguous().view(1,-1,4)
#print(pred_anchor_locs.shape)

pred_cls_scores = pred_cls_scores.permute(0,2,3,1).contiguous()
#print(pred_cls_scores.shape)

objectness_score = pred_cls_scores.view(1,50,50,9,2)[:,:,:,:,1].contiguous().view(1,-1)
#print(objectness_score.shape)

pred_cls_scores = pred_cls_scores.view(1,-1,2)
#print(pred_cls_scores.shape)

#pred_cls_scores和objectness_scores作为proposal层的输入，proposal层生成后续用于RoI网络的一系列proposal

nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()
objectness_score_numpy = objectness_score[0].data.numpy()

anc_heigth = anchor_base[:,2] - anchor_base[:,0]
anc_width = anchor_base[:,3] - anchor_base[:,1]
anc_ctr_y = anchor_base[:,0] + 0.5 * anc_heigth
anc_ctr_x = anchor_base[:,1] + 0.5 * anc_width

dy = pred_anchor_locs_numpy[:,0::4]
dx = pred_anchor_locs_numpy[:,1::4]
dh = pred_anchor_locs_numpy[:,2::4]
dw = pred_anchor_locs_numpy[:,3::4]

ctr_y = dy * anc_heigth[:,np.newaxis] + anc_ctr_y[:,np.newaxis]
ctr_x = dx * anc_width[:,np.newaxis] + anc_ctr_x[:,np.newaxis]
h = np.exp(dh) * anc_heigth[:,np.newaxis]
w = np.exp(dw) * anc_width[:,np.newaxis]

roi = np.zeros(pred_anchor_locs_numpy.shape,dtype=pred_anchor_locs_numpy.dtype)
roi[:,0::4] = ctr_y - 0.5 * h
roi[:,1::4] = ctr_x - 0.5 * w
roi[:,2::4] = ctr_y + 0.5 * h
roi[:,3::4] = ctr_x + 0.5 * w

img_size = (800,800)
roi[:, slice(0,4,2)] = np.clip(roi[:, slice(0,4,2)],0,img_size[0])
roi[:, slice(1,4,2)] = np.clip(roi[:, slice(1,4,2)],0,img_size[1])

#print(roi)

hs = roi[:,2] - roi[:,0]
ws = roi[:,3] - roi[:,1]
keep = np.where((hs >= min_size) & (ws >= min_size))[0]
roi = roi[keep,:]
score = objectness_score_numpy[keep]

#print(score.shape)

order = score.ravel().argsort()[::-1]
#print(order)

order = order[:n_train_pre_nms]
roi = roi[order,:]

#print(roi.shape)
#print(roi)

y1 = roi[:,0]
x1 = roi[:,1]
y2 = roi[:,2]
x2 = roi[:,3]

areas = (x2 - x1 + 1) * (y2 - y1 + 1)

score = score[order]
order = score.argsort()[::-1]
#print(area)
#area是一个数组
#print(order)


keep = []

while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i],x1[order[1:]])
    yy1 = np.maximum(y1[i],y1[order[1:]])
    xx2 = np.minimum(x2[i],x2[order[1:]])
    yy2 = np.minimum(y2[i],y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[i] + areas[order[1:]] - inter)

    inds = np.where(ovr <= nms_thresh)[0]
    order = order[inds + 1]

keep = keep[:n_train_post_nms]
roi = roi[keep]
#得到region proposals，作为fast r-cnn的输入
#print(roi.shape)
#print(roi)

#Proposal targets
n_sample = 128 
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5
neg_iou_thresh_lo = 0.0

ious = np.empty((len(roi),2), dtype=np.float32)
ious.fill(0)

for num1, i in enumerate(roi):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])

        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0.

        ious[num1, num2] = iou

#print(ious.shape)

gt_assignment = ious.argmax(axis=1)
max_iou = ious.max(axis=1)
#print(gt_assignment)
#print(max_iou)

gt_roi_label = labels[gt_assignment]
#print(gt_roi_label)

pos_roi_per_image = 32
pos_index = np.where(max_iou >= pos_iou_thresh)[0]
pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
if pos_index.size > 0:
    pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
#print(pos_index)

neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
neg_roi_per_this_image = n_sample - pos_roi_per_this_image
neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
if neg_index.size > 0 :
    neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
#print(neg_index)

keep_index = np.append(pos_index, neg_index)
gt_roi_labels = gt_roi_label[keep_index]
gt_roi_labels[pos_roi_per_this_image:] = 0
sample_roi = roi[keep_index]                                #预测框
#print(sample_roi.shape)

bbox_for_sample_roi = bbox[gt_assignment[keep_index]]       #每一个roi对应的目标框
#print(bbox_for_sample_roi.shape)

height = sample_roi[:,2] -sample_roi[:,0]
width = sample_roi[:,3] - sample_roi[:,1]
ctr_y = sample_roi[:,0] + 0.5 * height
ctr_x = sample_roi[:,1] + 0.5 * width
base_height = bbox_for_sample_roi[:,2] - bbox_for_sample_roi[:,0]
base_width = bbox_for_sample_roi[:,3] - bbox_for_sample_roi[:,1]
base_ctr_y = bbox_for_sample_roi[:,0] + 0.5 * base_height
base_ctr_x = bbox_for_sample_roi[:,1] + 0.5 * base_width

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width,eps)

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

gt_roi_locs = np.vstack((dy,dx,dh,dw)).transpose()      #预测框到目标框的变换
#print(gt_roi_locs.shape)

rois = torch.from_numpy(sample_roi).float()
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()
#print(rois.shape, roi_indices.shape)

indices_and_rois = torch.cat([roi_indices[:,None], rois], dim=1)
#print(indices_and_rois.shape)
xy_indices_and_rois = indices_and_rois[:,[0,2,1,4,3]]
indices_and_rois = xy_indices_and_rois.contiguous()
#print(xy_indices_and_rois.shape)


size = (7, 7)
adpative_max_pool = torch.nn.AdaptiveMaxPool2d(size[0], size[1])
output = []
rois = indices_and_rois.float()
rois[:, 1:].mul_(1/16.0)        #缩小以适应特征图
rois = rois.long()              #这一步是取整吧？
num_rois = rois.size(0)
'''
roi = rois[0]
im_idx = roi[0]
im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
adata = adpative_max_pool(im)[0].data
print(adata.shape)
'''

for i in range(num_rois):
    roi = rois[i]
    im_idx = roi[0]
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
    output.append(adpative_max_pool(im)[0].data)            #AdaptiveMaxPool2d的输出格式是啥？
output = torch.stack(output)
#print(output.shape)
output = torch.squeeze(output)      #去掉为1的维数
#print(output.size())
k = output.view(output.size(0), -1)
print(k.shape)


roi_haed_classifier = torch.nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096, 4096)])
cls_loc = torch.nn.Linear(4096, 21 * 4)
cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()
score = torch.nn.Linear(4096, 21)

k = torch.autograd.Variable(k)
k = roi_haed_classifier(k)
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)

print(roi_cls_loc.data.shape, roi_cls_score.data.shape)


# RPN 损失函数
print(pred_anchor_locs.shape)           # RPN网络预测的坐标系数
print(pred_cls_scores.shape)            # RPN网络预测的类别
print(anchor_locations.shape)           # anchor对应的实际坐标系数
print(anchor_labels.shape)              # anchor的实际类别，只有正负标签


rpn_loc = pred_anchor_locs[0]
rpn_score = pred_cls_scores[0]
gt_rpn_loc = torch.from_numpy(anchor_locations)
gt_rpn_score = torch.from_numpy(anchor_labels)          #gt_rpn_score其实是gt_rpn_label
print(rpn_loc.shape, rpn_score.shape, gt_rpn_loc.shape, gt_rpn_score.shape)

# 对于classification使用Cross Entropy损失函数
gt_rpn_score = torch.autograd.Variable(gt_rpn_score.long())
rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_score, gt_rpn_score, ignore_index=-1)
print(rpn_cls_loss)

# 对于Regression使用smooth L1损失
pos = gt_rpn_score.data > 0
mask = pos.unsqueeze(1).expand_as(rpn_loc)      #expand_as将pos变为与rpn_loc一样形状的tensor
print(mask.shape)

mask_loc_preds = rpn_loc[mask].view(-1,4)
mask_loc_targets = gt_rpn_loc[mask].view(-1,4)
print(mask_loc_preds.shape, mask_loc_targets.shape)

crit = torch.nn.SmoothL1Loss(reduction = 'sum')
rpn_loc_loss = crit(mask_loc_targets, mask_loc_preds)
print(rpn_loc_loss)
#x = np.abs(mask_loc_targets.numpy() - mask_loc_preds.data.numpy())
#print(x.shape)

#rpn_loc_loss = ((x < 1) * 0.5 * x**2) + ((x >= 1) * (x - 0.5))
#rpn_loc_loss = rpn_loc_loss.sum()
#print(rpn_loc_loss)

#N_reg = (gt_rpn_score > 0).float().sum()
#N_reg = np.squeeze(N_reg.data.numpy())
N_reg = torch.tensor(mask_loc_targets.size()[0], dtype=torch.float32)
print(N_reg)

print("N_reg: {}, {}".format(N_reg, N_reg.shape))
rpn_loc_loss = rpn_loc_loss / N_reg
#rpn_loc_loss = np.float32(rpn_loc_loss)

rpn_lambda = 10.
#rpn_cls_loss = np.squeeze(rpn_cls_loss.data.numpy())
print("rpn_cls_loss: {}".format(rpn_cls_loss))
print("rpn_loc_loss: {}".format(rpn_loc_loss))
rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
print("rpn_loss: {}".format(rpn_loss))


rpn_loss.backward()


# Fast R-CNN 损失函数
print(roi_cls_loc.shape)
print(roi_cls_score.shape)

print(gt_roi_locs.shape)
print(gt_roi_labels.shape)

gt_roi_loc = torch.from_numpy(gt_roi_locs)
gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()

print(gt_roi_loc.shape, gt_roi_label.shape)

gt_roi_label  = torch.autograd.Variable(gt_roi_label)
roi_cls_loss  = torch.nn.functional.cross_entropy(roi_cls_score, gt_roi_label, ignore_index=-1)
print(roi_cls_loss)

n_sample = roi_cls_loc.shape[0]
roi_loc = roi_cls_loc.view(n_sample, -1, 4)
print(roi_loc.shape)

roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]
print(roi_loc.shape)



