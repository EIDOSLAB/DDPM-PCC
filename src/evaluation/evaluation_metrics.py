"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""
import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm.auto import tqdm
from point_cloud_utils import pairwise_distances, sinkhorn

from ot.lp import emd2
import concurrent.futures

from chamferdist import ChamferDistance

# from evaluation.pointops.functions import pointops
from evaluation.psnr import get_psnr
from evaluation.density import get_density_metric

import open3d as o3d

from utils.dataset_modelnet import ModelNet40
from utils.dataset_shapenet import ShapeNetCore


import math
import sys

def earth_movers_distance(p, q, p_norm=2):
    # p: [N,3]
    # q: [2048,3]


    M = pairwise_distances(p.cpu(), q.cpu(), p_norm)
    a = np.ones(p.shape[0]) / p.shape[0]
    b = np.ones(q.shape[0]) / q.shape[0]

    if len(M.shape) == 1:
        M = np.expand_dims(M, axis = 0)
    
    try:
        res = emd2(a,b,M)
    except:
        print('error in emd2')
        print(f'p: {p.shape}')
        print(f'q: {q.shape}')
        print('----->')
        print(f'a: {a.shape}')
        print(f'b: {b.shape}')
        print(f'M: {M.shape}')
        sys.exit(1)

    return res


# Function to process a single batch
def process_batch(batch_index, tensor_p, tensor_q, p_norm=2):
    p = tensor_p[batch_index]
    q = tensor_q[batch_index]
    return earth_movers_distance(p, q, p_norm)

# def emd_approx(sample, ref):
    
#     emd = torch.zeros([sample.size(0)]).to(sample)
#     # d = 0
#     for i in range(sample.size(0)):
#         emd[i] = earth_movers_distance(sample[i],ref[i])
#     # return d/sample.size(0)
#     return emd

def emd_approx(sample, ref, eval_emd = True):
    batch_size = sample.shape[0]
    if not eval_emd:
        emd = torch.zeros([batch_size]).to(sample)
        return emd

    # Using ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, i, sample, ref) for i in range(batch_size)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return torch.tensor(results)


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def EMD_CD_non_batch(sample_pcs, ref_pcs, reduced=True, eval_emd = True, eval_cd = True, verbose = True):
    # sample_pcs, ref_pcs: [ p1, p2, .., pn ], each p = torch.tensor(N,3) where N can varies
    
    N_sample = len(sample_pcs)
    N_ref = len(ref_pcs)
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    iterator = range(0, N_sample, 1)
    if verbose:
        iterator = tqdm(iterator, desc='EMD-CD')

    cd_lst = []
    emd_lst = []
    chamferDist = ChamferDistance()
    for i in iterator:
        sample_pc = sample_pcs[i] # 1,N,3
        ref_pc = ref_pcs[i] # 1,2048,3

        if len(sample_pc.shape) == 2:
            sample_pc = sample_pc.unsqueeze(0)
        if len(ref_pc.shape) == 2:
            ref_pc = ref_pc.unsqueeze(0)

        # if i == 0:
        #     print('CD - EMD')
        #     print(sample_pc.shape)
        #     print(ref_pc.shape)
        #     print()

        if eval_cd:
            cd = chamferDist(sample_pc.float(), ref_pc.float(), bidirectional=True).detach().cpu().item()
        else:
            cd = 0.0
        cd_lst.append(cd)


        emd_batch = emd_approx(sample_pc, ref_pc, eval_emd = eval_emd)
        emd_lst.append(emd_batch)
    
    if reduced:
        cd = torch.tensor(cd_lst, dtype = torch.float).mean()
        emd = torch.tensor(emd_lst).mean()
    else:
        cd = torch.tensor(cd_lst)
        emd = torch.tensor(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results



def PSNR_non_batch(sample_pcs, ref_pcs, reduced=True, verbose = True, dataset = 'Modelnet'):
    N_sample = len(sample_pcs)
    N_ref = len(ref_pcs)
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    iterator = range(0, N_sample, 1)
    if verbose:
        iterator = tqdm(iterator, desc='PSNR-D2')


    peak = None
    if dataset == 'modelnet':
        print('\n\n!!! Eval Modelnet!! \n\n')
        test_loader = ModelNet40(
            path='/home/ids/gspadaro/repos/diffusion-point-cloud',
            num_points=2048,
            partition='test'
        )
    elif dataset == 'shapenet':
        print('\n\n!!! Eval Shapenet!! \n\n')
        test_loader = ShapeNetCore(
            path='/home/ids/gspadaro/repos/diffusion-point-cloud/data/shapenet.hdf5',
            cates=['all'],
            split='test',
            scale_mode='shape_unit'
        )
    else:
        raise NotImplementedError(f'dataset: {dataset} not yet implemented')
    
    psnr_list = []
    for i in iterator:

        sample_pc = sample_pcs[i] # 2048,3
        ref_pc = ref_pcs[i] # 2048,3

        if len(sample_pc.shape) == 3:
            assert sample_pc.shape[0] == 1
            sample_pc = sample_pc.squeeze(0)
        if len(ref_pc.shape) == 3:
            assert ref_pc.shape[0] == 1
            ref_pc = ref_pc.squeeze(0)

        # print(sample_pc.shape) # 2048,3
        # print(ref_pc.shape) # 2048,3

        ref_pc = ref_pc.cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ref_pc)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=12))
        pcd.normalize_normals()
        normals = torch.tensor(pcd.normals)
        ref_pc = torch.tensor(ref_pc)

        # print(type(ref_pc), ref_pc.shape) <class 'torch.Tensor'> torch.Size([2048, 3])
        # print(type(normals), normals.shape) <class 'torch.Tensor'> torch.Size([2048, 3])
        # print(type(sample_pc), sample_pc.shape) <class 'torch.Tensor'> torch.Size([2048, 3])


        ref_pc = ref_pc.unsqueeze(0).to('cuda')
        normals = normals.unsqueeze(0).to('cuda')
        sample_pc = sample_pc.unsqueeze(0).to('cuda')

        # if i == 0:
        #     print('PSNR')
        #     print(sample_pc.shape)
        #     print(ref_pc.shape)
        #     print(normals.shape)
        #     print()
        
        d2_psnr, peak = get_psnr(ref_pc, normals, sample_pc, test_loader, peak)

        if not math.isinf(d2_psnr.item()):
            # print(peak)
            psnr_list.append(d2_psnr.item())
        else:
            print('inf value founded..')
    
    if reduced:
        psnr = torch.tensor(psnr_list).mean()
    else:
        psnr = torch.tensor(psnr_list)

    results = {
        'PSNR-D2': psnr,
    }
    return results



def Density_non_batch(sample_pcs, ref_pcs, reduced=True, verbose = True):
    N_sample = len(sample_pcs)
    N_ref = len(ref_pcs)
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    iterator = range(0, N_sample, 1)
    if verbose:
        iterator = tqdm(iterator, desc='Density')

    density_lst = []
    for i in iterator:
        sample_pc = sample_pcs[i] # 1,N,3
        ref_pc = ref_pcs[i] # 1,2048,3

        if len(sample_pc.shape) == 2:
            sample_pc = sample_pc.unsqueeze(0)
        if len(ref_pc.shape) == 2:
            ref_pc = ref_pc.unsqueeze(0)

        ref_pc = ref_pc.to('cuda')
        sample_pc = sample_pc.to('cuda')

        # if i == 0:
        #     print('DENSITY')
        #     print(sample_pc.shape)
        #     print(ref_pc.shape)
        #     print()

        density = get_density_metric(ref_pc, sample_pc).cpu().item()
        density_lst.append(density)

    
    if reduced:
        density = torch.tensor(density_lst).mean()
    else:
        density = torch.tensor(density_lst)

    results = {
        'Density': density,
    }
    return results




def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True, eval_emd = True, verbose = True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc='EMD-CD')

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr = distChamfer(sample_batch, ref_batch)
        cd = dl.mean(dim=1) + dr.mean(dim=1)
        cd_lst.append(cd)

        emd_batch = emd_approx(sample_batch, ref_batch, eval_emd = eval_emd)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, verbose=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc='Pairwise EMD-CD')
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat([
        torch.cat((Mxx, Mxy), 1),
        torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def lgan_mmd_cov_match(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }, min_idx.view(-1)


def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    ## CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })
    
    ## EMD
    # res_emd = lgan_mmd_cov(M_rs_emd.t())
    # results.update({
    #     "%s-EMD" % k: v for k, v in res_emd.items()
    # })

    for k, v in results.items():
        print('[%s] %.8f' % (k, v.item()))

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    ## CD
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    ## EMD
    # one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    # results.update({
    #     "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    # })

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube. If clip_sphere it True
    it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(
        sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(
        sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(
        ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(
        pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in tqdm(pclouds, desc='JSD'):
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


if __name__ == '__main__':
    # a = torch.randn([16, 2048, 3]).cuda()
    # b = torch.randn([16, 2048, 3]).cuda()

    a = torch.randn([16, 3, 2048]).cuda()
    b = torch.randn([16, 3, 2048]).cuda()

    print(LPIPS(a, b, batch_size=8, reduced=False))
    