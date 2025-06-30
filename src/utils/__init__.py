from utils.transform import RandomRotate
from utils.dataset_shapenet import ShapeNetCore
from utils.dataset_modelnet import ModelNet40
from utils.data import get_data_iterator
from torch.utils.data import DataLoader


# import compressai.transforms as transforms
# from utils.modelnet import ModelNetDataset
import torch
from torchvision.transforms import Compose

import timm


def get_modelnet(args, timm_args = False):
    transform = None
    if args.rotate:
        transform = RandomRotate(180, ['pointcloud'], axis=1)
    print('Transform: %s' % repr(transform))

    print('Loading datasets... (ModelNet)')
    train_dset = ModelNet40(
        path=args.dataset_path,
        num_points=2048,
        partition='train',
        transform=transform
    )
    test_dset = ModelNet40(
        path=args.dataset_path,
        num_points=2048,
        partition='test'
    )

    if not timm_args:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dset, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dset)
            test_sampler = torch.utils.data.SequentialSampler(test_dset)
    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset)
            test_sampler = timm.data.distributed_sampler.OrderedDistributedSampler(test_dset)
        else:
            train_sampler = None #torch.utils.data.RandomSampler(train_dset)
            test_sampler = None #torch.utils.data.SequentialSampler(test_dset)


    train_loader = DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
        sampler=train_sampler
    )
    train_iterator = get_data_iterator(train_loader)

    val_loader = None #DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0, sampler=val_sampler)
    test_loader = DataLoader(test_dset, batch_size=args.val_batch_size, num_workers=0, sampler=test_sampler)

    return train_iterator, train_loader, val_loader, test_loader, train_sampler


def get_shapenet(args, timm_args = False):
    transform = None
    if args.rotate:
        transform = RandomRotate(180, ['pointcloud'], axis=1)
    print('Transform: %s' % repr(transform))


    print('Loading datasets...')
    train_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='train',
        scale_mode=args.scale_mode,
        transform=transform,
    )
    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='val',
        scale_mode=args.scale_mode
    )
    test_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.scale_mode
    )

    if not timm_args:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dset, shuffle=False)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dset, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dset)
            test_sampler = torch.utils.data.SequentialSampler(test_dset)
            val_sampler = torch.utils.data.SequentialSampler(val_dset)
    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset)
            test_sampler = timm.data.distributed_sampler.OrderedDistributedSampler(test_dset)
            val_sampler = timm.data.distributed_sampler.OrderedDistributedSampler(val_dset)
        else:
            train_sampler = None #torch.utils.data.RandomSampler(train_dset)
            test_sampler = None #torch.utils.data.SequentialSampler(test_dset)
            val_sampler = None #torch.utils.data.SequentialSampler(val_dset)





    train_loader = DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
        sampler=train_sampler
    )
    train_iterator = get_data_iterator(train_loader)

    val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0, sampler=val_sampler)
    test_loader = DataLoader(test_dset, batch_size=args.val_batch_size, num_workers=0, sampler=test_sampler)

    return train_iterator, train_loader, val_loader, test_loader, train_sampler



if __name__ == '__main__':

    from opt import get_args

    args = get_args()
    args.val_batch_size = 1
    args.train_batch_size = 1
    args.distributed = False
    # args.dataset_path = '/home/ids/gspadaro/repos/diffusion-point-cloud/data/shapenet.hdf5'
    args.dataset_path = '/home/ids/gspadaro/repos/diffusion-point-cloud/'
    
    # train_iterator, train_loader, val_loader, test_loader, train_sampler = get_shapenet(args, timm_args=False)
    # Total ShapeNet: 51.127 -> train 43433 + test: 7694

    train_iterator, train_loader, _, test_loader, train_sampler = get_modelnet(args)
    # Total ModelNet: 12308 -> train 9840 + test: 2468
    # train_iterator,train_loader, val_loader, test_loader = get_dataloaders(args)


    print(len(train_loader))
    print(len(test_loader))


    batch = next(train_iterator)
    x = batch['pointcloud']
    # # n = batch['normal']
    # x = batch['pointcloud']


    print(x.shape) # B,2048,3
    # print(torch.min(x), torch.max(x))

    # shift = batch['shift']
    # scale = batch['scale']

    # x = x * scale + shift
    # print(torch.min(x), torch.max(x))


    # print(n.shape)
    # print(torch.min(n), torch.max(n))