import torch
import torch.nn as nn
import torchvision
import numpy as np 

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.features = self._vgg_layers(cfg)
        self._rpn_layers()
        size = (7, 7)
        self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(size[0], size[1])
        self.roi_classifier()

    def _vgg_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x ,kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)
                        ]
                in_channels = x
            
        return nn.Sequential(*layers)

    def _rpn_layers(self, mid_channels=512, in_channels=512, n_anchor=9):
        self.rpn_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        self.rpn_conv.weight.data.normal_(0, 0.01)
        self.rpn_conv.bias.data.zero_()

        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()

        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()

    def roi_classifier(self, class_num=20):
        self.roi_classifier = nn.Sequential(
            *[nn.Linear(25088, 4096),
              nn.ReLU(),
              nn.Linear(4096, 4096),
              nn.ReLU()])
        self.cls_loc = nn.Linear(4096, (class_num+1) * 4)
        self.cls_loc.weight.data.normal_(0, 0.01)
        self.cls_loc.bias.data.zero_()

        self.score = nn.Linear(4096, class_num+1)

    def rpn_loss(self, rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_label,rpn_lambda=10.0):
        gt_rpn_label = torch.autograd.Variable(gt_rpn_label.long())
        rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

        pos = gt_rpn_label > 0
        mask = pos.unsqueeze(1).expand_as(rpn_loc)

        mask_loc_preds = rpn_loc[mask].view(-1,4)
        mask_loc_targets = gt_rpn_loc[mask].view(-1,4)

        crit = torch.nn.SmoothL1Loss()
        rpn_loc_loss = crit(mask_loc_targets, mask_loc_preds) * 4       #需要乘以4，是因为crit函数求得的均值是每一个坐标的，我们求的均值是一个框的

        rpn_loss = rpn_cls_loss + rpn_lambda * rpn_loc_loss
        
        return rpn_loss
    
    def roi_loss(self, roi_loc, roi_score, target_loc, target_label, roi_lambda=10.0):
        target_label = torch.autograd.Variable(target_label.long())
        roi_cls_loss = torch.nn.functional.cross_entropy(roi_score, target_label, ignore_index=-1)

        pos = target_label.data > 0
        mask = pos.unsqueeze(1).expand_as(roi_loc)

        mask_roi_loc = roi_loc[mask].view(-1,4)
        mask_target_loc = target_loc[mask].view(-1,4)

        crit = torch.nn.SmoothL1Loss()
        roi_loc_loss = crit(mask_target_loc, mask_roi_loc) * 4

        roi_loss = roi_cls_loss + roi_lambda * roi_loc_loss

        return roi_loss

    def forward(self, data):
        out_map = self.features(data)
        x = self.rpn_conv(out_map)
        rpn_anchor_locs = self.reg_layer(x)
        rpn_cls_scores = self.cls_layer(x)

        return out_map, rpn_anchor_locs, rpn_cls_scores
    

    



        