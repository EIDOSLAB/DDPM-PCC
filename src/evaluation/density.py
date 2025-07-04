import torch
import numpy as np
# from evaluation.functions.pointops.functions import pointops
from evaluation.pointops.functions import pointops

from torch import nn
from einops import rearrange, repeat
from evaluation.utils import index_points

# Script taken from https://github.com/yunhe20/D-PCC/blob/master/metrics/density.py




def get_local_density(xyzs, density_radius = 0.15):
    # input: (b, n, 3)

    # (b, n, 100), assume the maximum number of neighbors within radius is 100
    knn_idx = pointops.knnquery_heap(100, xyzs, xyzs).long()
    # (b, 3, n)
    xyzs_trans = xyzs.permute(0, 2, 1).contiguous()
    # (b, 3, n, 100)
    knn_xyzs = index_points(xyzs_trans, knn_idx)
    # (b, 3, n, 100)
    repeated_xyzs = repeat(xyzs_trans, 'b c n -> b c n k', k=100)
    # (b, n, 100)
    pairwise_distance = torch.norm(knn_xyzs - repeated_xyzs, dim=1, p=2)

    # mask those neighbors whose distances are larger than radius
    # density-radius -> 0.15
    dist = torch.where(pairwise_distance<=density_radius, pairwise_distance, torch.tensor([0]).cuda().float())
    # (b, n)
    dist = torch.sum(dist, dim=-1)

    # get the neighbor number within radius
    num = torch.where(pairwise_distance<=density_radius, torch.tensor([1]).cuda().float(), torch.tensor([0]).cuda().float())
    # (b, n)
    num = torch.sum(num, dim=-1)

    # mean distance
    dist = dist / num

    return dist, num




def get_density_diff(xyzs1, dist1, num1, xyzs2, dist2, num2, dist_coe = 1e-4):
    # xyzs: (b, n, 3) dist and num: (b, n)

    eps = 1e-8
    # (n)
    nearest_idx = pointops.knnquery_heap(1, xyzs2, xyzs1).view(-1).long()
    nearest_num = num2[:, nearest_idx] + eps
    nearest_dist = dist2[:, nearest_idx] + eps

    # print(nearest_num.shape)
    # print(num1.shape)


    num_diff = torch.abs(nearest_num - num1) / nearest_num
    # dist_coe -> 1e-5
    dist_diff = dist_coe * (torch.abs(nearest_dist - dist1) / nearest_dist)

    num_diff = num_diff.mean()
    dist_diff = dist_diff.mean()

    return num_diff + dist_diff




def get_density_metric(gt_xyzs, pred_xyzs):
    # input: (b, n, 3)
    assert gt_xyzs.shape[0] == 1
    assert pred_xyzs.shape[0] == 1


    torch.cuda.empty_cache()
    # (b, n)
    gt_dist, gt_num = get_local_density(gt_xyzs)
    torch.cuda.empty_cache()
    pred_dist, pred_num = get_local_density(pred_xyzs)

    gt2pred_density_diff = get_density_diff(gt_xyzs, gt_dist, gt_num, pred_xyzs, pred_dist, pred_num)
    pred2gt_density_diff = get_density_diff(pred_xyzs, pred_dist, pred_num, gt_xyzs, gt_dist, gt_num)
    density_metric = gt2pred_density_diff + pred2gt_density_diff

    return density_metric

if __name__ == '__main__':
    a = torch.rand([1, 2048, 3]).cuda()
    b = torch.rand([1, 2048, 3]).cuda()
    print(get_density_metric(a, b).item())


