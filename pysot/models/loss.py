# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from torch import nn

import torch
import torch.nn.functional as F


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    label=label.long()
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero(as_tuple =False).squeeze().cuda()
    neg = label.data.eq(0).nonzero(as_tuple =False).squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

            
class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        
        pred_left = pred[:,:, 0]
        pred_top = pred[:,:, 1]
        pred_right = pred[:,:, 2]
        pred_bottom = pred[:,:, 3]

        target_left = target[:,:, 0]
        target_top = target[:,:, 1]
        target_right = target[:,:, 2]
        target_bottom = target[:,:, 3]
        
        x1 = torch.min(pred_left, pred_right)
        y1 = torch.min(pred_top, pred_bottom)
        x2 = torch.max(pred_left, pred_right)
        y2 = torch.max(pred_top, pred_bottom)
    
        xkis1 = torch.max(x1, target_left)
        ykis1 = torch.max(y1, target_top)
        xkis2 = torch.min(x2, target_right)
        ykis2 = torch.min(y2, target_bottom)
    
        xc1 = torch.min(x1, target_left)
        yc1 = torch.min(y1, target_top)
        xc2 = torch.max(x2, target_right)
        yc2 = torch.max(y2, target_bottom)
    
        intsctk = torch.zeros(x1.size()).cuda()
        
        mask = (ykis2 > ykis1) * (xkis2 > xkis1)
        intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
        unionk = (x2 - x1) * (y2 - y1) + (target_right - target_left) * (target_bottom - target_top) - intsctk + 1e-7
        iouk = intsctk / unionk
    
        area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
        miouk = iouk - ((area_c - unionk) / area_c)

        losses = 1-miouk
        weight=weight.view(losses.size())
        if weight.sum()>0:
            
            return (losses * weight).sum() / (weight.sum()+1e-6)
        else:
            return (losses *weight).sum()
            
