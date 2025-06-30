import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import scatter
import numpy as np
import random

class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_instance_norm = False, legacy = False):
        super().__init__()
        self.zdim = zdim
        self.legacy = legacy
        if use_instance_norm:
            norm_fz =  nn.InstanceNorm1d
        else:
            norm_fz = nn.BatchNorm1d

        if legacy:
            self.conv1 = nn.Conv1d(input_dim, 128, 1)
            self.conv2 = nn.Conv1d(128, 128, 1)
            self.conv3 = nn.Conv1d(128, 256, 1)
            self.conv4 = nn.Conv1d(256, 512, 1)
        else:
            self.conv1 = nn.Linear(input_dim, 128)
            self.conv2 = nn.Linear(128, 128)
            self.conv3 = nn.Linear(128, 256)
            self.conv4 = nn.Linear(256, 512)
        self.bn1 = norm_fz(128)
        self.bn2 = norm_fz(128)
        self.bn3 = norm_fz(256)
        self.bn4 = norm_fz(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = norm_fz(256)
        self.fc_bn2_m = norm_fz(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = norm_fz(256)
        self.fc_bn2_v = norm_fz(128)

    def forward(self, x, batch_idxs = None):
        # x: B,N,3 or (B*N, 3)

        if self.legacy:
            x = x.permute(0,2,1)

        if len(x.shape) == 3 and not self.legacy:
            B,N,C = x.shape
            if batch_idxs is None:
                batch_idxs = torch.arange(B)
                batch_idxs = batch_idxs.repeat_interleave(N).to(x.device) # (B*N)
            x = x.reshape(-1,C) # (B*N, 3)
        


        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        # MAX POOLING
        if batch_idxs is None:
            x = torch.max(x, 2, keepdim=True)[0]
            # x: B,512,1
            x = x.view(-1, 512)
            # x: B,512
        else:

            # x: B*N,512,1

            x = scatter(x,batch_idxs, reduce = 'max')
            # batch_size = batch_idxs.max() + 1
            # x = torch.zeros((batch_size, x.size(1))).to(x.device).scatter_reduce(0, batch_idxs.unsqueeze(1).expand(-1, x.size(1)), x, reduce='amax')
            # x: B,512
        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        # # x: B,256
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        # v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        # v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        # v = self.fc3_v(v)
        v = None

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v



if __name__ == '__main__':
    device = 'cuda'
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    batch_size = 8
    # code = torch.rand((batch_size, 256))
    x = torch.rand((batch_size,5,3))

    x = x.view(-1,1,3)

    idxs = torch.arange(batch_size)
    idxs = idxs.repeat_interleave(5)

    encoder = PointNetEncoder(zdim=256, use_instance_norm=True) #.to(device)
    encoder.train()

    x, _ = encoder(x, batch_idxs = idxs)
    print(x.shape)
    
    torch.save(x,'out_encoder_flat.pth')

