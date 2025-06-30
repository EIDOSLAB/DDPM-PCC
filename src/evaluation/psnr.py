import torch
import os
from evaluation.pointops.functions import pointops

from evaluation.utils import index_points
# Script taken from https://github.com/yunhe20/D-PCC/blob/master/metrics/PSNR.py
import numpy as np
import open3d as o3d



def get_peak(test_loader):
    peak = 0
    # print('Eval peak..')

    for i, test_dict in enumerate(test_loader):
        # (n, 3)
        xyzs = test_dict['pointcloud'].float().cuda()
        shift = test_dict['shift'].float().cuda()
        scale = test_dict['scale'].float().cuda()
        xyzs = xyzs * scale + shift

        # print(xyzs.shape)
        # (1, n, 3)
        xyzs = xyzs.unsqueeze(0)


        # exclude itself, (b, n)
        idx = pointops.knnquery_heap(2, xyzs, xyzs)[..., 1].long()
        # (b, 3, n)
        xyzs_trans = xyzs.permute(0, 2, 1).contiguous()
        # (b, 3, n)
        nearest_xyzs_trans = index_points(xyzs_trans, idx)
        # (b, n, 3)
        nearest_xyzs = nearest_xyzs_trans.permute(0, 2, 1).contiguous()
        # max distance of the current batch
        max_dist = torch.norm(nearest_xyzs - xyzs, dim=-1).max()
        # max distance of the whole dataset
        peak = max_dist.item() if max_dist.item() > peak else peak

    return peak




def sum_d2(p1, p2, normal):
    # x: (n, 3)
    return torch.sum(torch.sum((p1 - p2) * normal, dim=1) ** 2)


def psnr(peak, mse):
    max_energy = 3*peak*peak
    return 10 * torch.log10(max_energy / mse)


def get_psnr(gt_xyzs, gt_normals, pred_xyzs, test_loader, peak = None):
    # gt: (1, n, 3) pred: (1, m, 3)

    # if args.peak == None:
        # get_peak(test_loader, args)
    # peak = args.peak
    if peak is None:
        peak = get_peak(test_loader)


    # (n)
    gt2pred_idx = pointops.knnquery_heap(1, pred_xyzs, gt_xyzs).view(-1).long()
    # (m)
    pred2gt_idx = pointops.knnquery_heap(1, gt_xyzs, pred_xyzs).view(-1).long()

    # (n, 3)
    gt_xyzs = gt_xyzs.squeeze(0)
    gt_normals = gt_normals.squeeze(0)
    # (m, 3)
    pred_xyzs = pred_xyzs.squeeze(0)

    # (n, 3)
    gt_nearest_xyzs = pred_xyzs[gt2pred_idx, :]
    # (m, 3)
    pred_nearest_xyzs = gt_xyzs[pred2gt_idx, :]
    # (m, 3)
    pred_normals = gt_normals[pred2gt_idx, :]

    # (n, 3)
    gt_nearest_normals = pred_normals[gt2pred_idx, :]
    # (m, 3)
    pred_nearest_normals = gt_normals[pred2gt_idx, :]

    # calculate the psnr
    d2_sum_gt2pred = sum_d2(gt_xyzs, gt_nearest_xyzs, gt_nearest_normals)
    d2_sum_pred2gt = sum_d2(pred_xyzs, pred_nearest_xyzs, pred_nearest_normals)
    d2_mse_gt2pred = d2_sum_gt2pred / gt_xyzs.shape[0]
    d2_mse_pred2gt = d2_sum_pred2gt / pred_xyzs.shape[0]
    d2_max_mse = max(d2_mse_gt2pred, d2_mse_pred2gt)
    d2_psnr = psnr(peak, d2_max_mse)

    return d2_psnr, peak

