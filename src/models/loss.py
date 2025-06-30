import torch.nn as nn
import torch
import torch.nn.functional as F
import math


def cal_loss(pred, gold, vq_loss, alpha = 10):
    rec_loss = F.mse_loss(pred, gold, reduction='mean')
    return rec_loss + alpha*vq_loss, rec_loss


class DistortionVQLoss(nn.Module):

    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha


    def forward(self, pred, gold, vq_loss):
        out = {}

        out['vq_loss'] = vq_loss
        out['mse_loss'] = F.mse_loss(pred, gold, reduction='mean') 

        out['loss'] = out['mse_loss'] + self.alpha*out['vq_loss']
        return out