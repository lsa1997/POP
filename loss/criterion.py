from unicodedata import normalize
import torch.nn as nn
import torch
from torch.nn import functional as F

class CELoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, pred, target, aux_pred=None):
        '''
            pred      : [BxKxhxw]
            target    : [BxHxW]
        '''
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)
        main_loss = self.seg_criterion(scale_pred, target)
        if aux_pred is not None:
            scale_aux_pred = F.interpolate(input=aux_pred, size=(h, w), mode='bilinear', align_corners=True)
            aux_loss = self.seg_criterion(scale_aux_pred, target)
            total_loss = main_loss + 0.4 * aux_loss
            loss_dict = {'total_loss':total_loss, 'main_loss':main_loss, 'aux_loss':aux_loss}
        else:
            loss_dict = {'total_loss':main_loss}
        return loss_dict

class OrthLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(OrthLoss, self).__init__()
        self.ignore_index = ignore_index
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.w = 10.0

    def get_orth_loss(self, proto_sim, is_ft=False):
        '''
            protos:   : [K1xK2] K1 <= K2
        '''
        eye_sim = torch.triu(torch.ones_like(proto_sim), diagonal=1)
        loss_orth = torch.abs(proto_sim[eye_sim == 1]).mean()
        return loss_orth

    def forward(self, preds, target, is_ft=False, proto_sim=None, aux_preds=None):
        '''
            pred      : [BxKaxhxw]
            target    : [BxHxW]
            proto_sim:   : [K1xK2]
        '''
        scale_pre = F.interpolate(input=preds, size=target.shape[1:], mode='bilinear', align_corners=True)
        seg_loss = self.seg_criterion(scale_pre, target)

        orth_loss = self.get_orth_loss(proto_sim, is_ft=is_ft)

        if aux_preds is not None:
            scale_aux_pred = F.interpolate(input=aux_preds, size=target.shape[1:], mode='bilinear', align_corners=True)
            aux_loss = self.seg_criterion(scale_aux_pred, target)
            total_loss = seg_loss + orth_loss * self.w + 0.4 * aux_loss
            loss_dict = {'total_loss':total_loss, 'seg_loss':seg_loss, 'aux_loss':aux_loss, 'orth_loss':orth_loss}
        else:
            total_loss = seg_loss + orth_loss * self.w
            loss_dict = {'total_loss':total_loss, 'seg_loss':seg_loss, 'orth_loss':orth_loss}

        return loss_dict
