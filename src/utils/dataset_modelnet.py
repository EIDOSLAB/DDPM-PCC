#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import sys
from torch.utils.data import DataLoader
import random
from copy import copy


def download(path):
    DATA_DIR = os.path.join(path, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(path, partition):
    download(path)
    DATA_DIR = os.path.join(path, 'data')
    # all_data = []
    all_label = []
    pointclouds = []
    # i = 0
    pc_id = 0
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = torch.from_numpy(f['data'][:].astype('float32'))
        # label = f['label'][:].astype('int64')

        # if i == 0:
        #     print(data.shape)
        #     print(torch.min(data), torch.max(data))
        #     print(f'mean: {data.mean(dim=0)}')   
        #     print(f'std: {data.std(dim=0)}')
        # i += 1
        for pc in data:
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
            pc = (pc - shift) / scale

            pointclouds.append({
                'pointcloud': pc,
                'shift': shift,
                'scale': scale,
                'id': pc_id
            })
            pc_id += 1

        f.close()
        # all_data.append(data)
        # all_label.append(label)
    # all_data = torch.cat(all_data, axis=0)
    # print(all_data.shape)
    # print()

    # all_label = np.concatenate(all_label, axis=0)
    pointclouds.sort(key=lambda data: data['id'], reverse=False)
    random.Random(2020).shuffle(pointclouds)
    return pointclouds #, all_label


# def translate_pointcloud(pointcloud):
#     xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
#     xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
#     translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
#     return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, path, num_points, partition='train', transform = None):
        self.pointclouds = load_data(path, partition)
        self.num_points = num_points
        self.partition = partition     
        self.transform = transform   

    # def __getitem__(self, item):
    #     pointcloud = self.data[item][:self.num_points]
    #     # label = self.label[item]
    #     # if self.partition == 'train':
    #     #     # pointcloud = translate_pointcloud(pointcloud)
    #     #     np.random.shuffle(pointcloud)

    #     if self.transform is not None:
    #         pointcloud = self.transform(pointcloud)
    #     return pointcloud

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data


    def __len__(self):
        # return self.data.shape[0]
        return len(self.pointclouds)
    

