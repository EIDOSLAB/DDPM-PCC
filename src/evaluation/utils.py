import torch
from einops import rearrange

def index_points(xyzs, idx):
    """
    Input:
        xyzs: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = xyzs.shape[1]

    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)

    # (b, c, (s k))
    res = torch.gather(xyzs, 2, idx[:, None].repeat(1, fdim, 1))

    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res
