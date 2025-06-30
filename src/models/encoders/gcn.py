#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from models.gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDynBlock2d, DilatedKnnGraph, PlainDynBlock2d


class DeepGCN(torch.nn.Module):
    def __init__(self, 
                 channels = 64,
                 k = 16,
                 act = 'relu',
                 norm = 'batch',
                 bias = True,
                 epsilon = 0.2,
                 stochastic = False,
                 conv = 'mr',
                 emb_dims = 256,
                 n_blocks = 7,
                 in_channels = 3,
                 block = 'res',
                 use_dilation = True):
        
        super(DeepGCN, self).__init__()

        self.n_blocks = n_blocks


        knn = 'matrix'  # implement knn using matrix multiplication
        c_growth = channels

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(in_channels, channels, conv, act, norm, bias=False)

        if block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon, knn)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks-1)) * self.n_blocks // 2)

        elif block.lower() == 'res':
            if use_dilation:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, i + 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn)
                                      for i in range(self.n_blocks - 1)])
            else:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn)
                                      for _ in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        else:
            # Plain GCN. No dilation, no stochastic, no residual connections
            stochastic = False

            self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm,
                                                  bias, stochastic, epsilon, knn)
                                  for i in range(self.n_blocks - 1)])

            fusion_dims = int(channels+c_growth*(self.n_blocks-1))

        self.fusion_block = BasicConv([fusion_dims, emb_dims], 'leakyrelu', norm, bias=False)
        self.merge = BasicConv([emb_dims * 2, emb_dims], 'leakyrelu', norm)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        # B,N,C
        inputs = inputs.permute(0,2,1).unsqueeze(-1)
        # print(f'input: {inputs.shape}')
        # print('\n')
        feats = [self.head(inputs, self.knn(inputs))]
        # print('After heads')
        # print([f.shape for f in feats])
        # print('\n')
        
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
            # print(f'Block {i}')
            # print([f.shape for f in feats])
            # print('\n')

        feats = torch.cat(feats, dim=1)
        # print(f'cat: {feats.shape}')
        # print('\n')
    
        fusion = self.fusion_block(feats)
        # print(f'fusion: {fusion.shape}')
        # print('\n')
        
        x1 = F.adaptive_max_pool2d(fusion, 1)
        # print(f'x1: {x1.shape}')
        # print('\n')
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        return self.merge(torch.cat((x1, x2), dim=1)).squeeze(-1).squeeze(-1), None


if __name__ == '__main__':
    device = torch.device('cuda')

    # feats = torch.rand((8, 3, 1024, 1), dtype=torch.float).to(device)
    feats = torch.rand((8,1024,3)).to(device)

    print('Input size {}'.format(feats.size()))
    net = DeepGCN().to(device)
    out = net(feats)
    print(f'Output: {out.shape}')
    # print(net)
    # print('Output size {}'.format(out.size()))
