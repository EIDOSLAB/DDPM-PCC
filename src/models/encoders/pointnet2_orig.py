import torch.nn as nn
import torch.nn.functional as F
from models.encoders.pointnet2_utils import PointNetSetAbstraction

import torch

class PointNetOrig2Encoder(nn.Module):
    def __init__(self,z_dim,normal_channel=False, mode = 0, use_dropout = True):
        super(PointNetOrig2Encoder, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        if mode == 0:
            # original setting:
            mlp1 = [64, 64, 128]
            mlp2 = [128, 128, 256]
            mlp3 = [256, 512, 1024]

            self.fc1 = nn.Linear(1024, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.drop1 = nn.Dropout(0.4) if use_dropout else nn.Identity()
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.drop2 = nn.Dropout(0.4) if use_dropout else nn.Identity()
            self.fc3 = nn.Linear(256, z_dim)

        
        elif mode == 1:
            # pointnet setting:
            mlp1 = [128]
            mlp2 = [256]
            mlp3 = [512]

            self.fc1 = nn.Linear(512, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.drop1 = nn.Dropout(0.4) if use_dropout else nn.Identity() 
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.drop2 = nn.Dropout(0.4) if use_dropout else nn.Identity()
            self.fc3 = nn.Linear(128, z_dim)

        elif mode == 2:
            # easy original:
            mlp1 = [64, 128]
            mlp2 = [128, 256]
            mlp3 = [256, 512, 1024]

            self.fc1 = nn.Linear(1024, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.drop1 = nn.Dropout(0.4) if use_dropout else nn.Identity()
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.drop2 = nn.Dropout(0.4) if use_dropout else nn.Identity()
            self.fc3 = nn.Linear(256, z_dim)


        elif mode == 3:
            # mix:
            mlp1 = [64, 128]
            mlp2 = [128, 256]
            mlp3 = [256, 512]

            self.fc1 = nn.Linear(512, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.drop1 = nn.Dropout(0.4) if use_dropout else nn.Identity()
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.drop2 = nn.Dropout(0.4) if use_dropout else nn.Identity()
            self.fc3 = nn.Linear(128, z_dim)



        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=in_channel, 
            mlp=mlp1, 
            group_all=False)
        
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3, 
            mlp=mlp2, 
            group_all=False)
        
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, 
            mlp=mlp3, 
            group_all=True)
        
        self.bottleneck_size = mlp3[-1]
        
        

    def forward(self, xyz, batch_idxs = None):
        # B,N,C
        xyz = xyz.permute(0,2,1)
        
        B, _, _ = xyz.shape

        # print(f'Input: {xyz.shape}')
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        # print(f'sa1: {l1_xyz.shape}, {l1_points.shape}')

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(f'sa2: {l2_xyz.shape}, {l2_points.shape}')

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(f'sa3: {l3_xyz.shape}, {l3_points.shape}')

        
        x = l3_points.view(B, self.bottleneck_size)
        # print(f'reshape: {x.shape}')

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        return x, None #, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
    


if __name__ == '__main__':
    device = 'cuda'
    
    x = torch.rand((128,2048,3)).to(device)
    encoder = PointNetOrig2Encoder(z_dim=256, mode=3, use_dropout = False).to(device)

    print(encoder)
    out,_ = encoder(x)
    print('\n-----------------')
    print(f'out: {out.shape}')