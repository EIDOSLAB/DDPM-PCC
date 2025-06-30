import os
import sys 

import torch


from utils.misc import seed_all, get_linear_scheduler, CheckpointManager, BlackHole
from models.autoencoder import AutoEncoder
from models.quantized_ae import AutoEncoderVQ
from opt import get_args

from engine import train, validate_loss, validate_inspect
from utils import get_modelnet, get_shapenet
import wandb

from utils_ddp import CustomDataParallel

import logging
from timm.utils.log import setup_default_logging
from models.loss import DistortionVQLoss

import torch.nn.functional as F


def cal_loss(pred, gold, qv_loss = None):
    loss = F.mse_loss(pred, gold, reduction='mean')

    # loss = torch.mean((pred - gold) ** 2)
    return loss

def main():
    log_wandb = False


    args = get_args()
    print('Running job..')
    compress = args.dim_codecs > 0


    args.distributed = False

    print('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')

    if compress:
        args.save_dir = f'{args.save_dir}_latent{args.latent_dim}_steps{args.num_steps}_rotate{args.rotate}_vqalpha{args.vq_alpha}_num_codecs{args.num_codecs}_dim_codecs{args.dim_codecs}'
    else:
        args.save_dir = f'{args.save_dir}_latent{args.latent_dim}_steps{args.num_steps}_rotate{args.rotate}'


    if log_wandb:
        wandb.login()

        wandb.init(
            # Set the project where this run will be logged
            project='PointCloud_Compression', 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name = f'{args.save_dir}',
            config = vars(args))
    
    seed_all(args.seed)

    best_cd_val = float('inf')

    if args.logging:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f'Results will be saved in {args.save_dir}')
        ckpt_mgr = CheckpointManager(args.save_dir)
    else:
        ckpt_mgr = BlackHole

    # Logging
    print(args)

    # Datasets and loaders
    pyg_ds = False
    if args.dataset == 'shapenet':
        train_iterator, _, val_loader, test_loader, train_sampler = get_shapenet(args, timm_args=False)
    elif args.dataset == 'modelnet':
        train_iterator, _, _, test_loader, train_sampler = get_modelnet(args)
        val_loader = test_loader
    else:
        raise NotImplementedError('dataset non implemeted')

        


    # Model
    print('Building model...')
    start_iter = 1
    
    args_model = args
    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume)
        args_model = ckpt['args']

    device = torch.device("cuda")

    if not compress:
        print('Using Model w/out vector quant.')
        model = AutoEncoder(args_model).to(device)
    else:
        model = AutoEncoderVQ(args_model).to(device)

    # if args.distributed:
    if args.num_gpu > 1:
        print('Using DataParallel')
        # model = torch.nn.DataParallel(model)
        model = CustomDataParallel(model)

        print("Let's use", torch.cuda.device_count(), "GPUs!")


    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr
    )

    if ckpt is not None:
        print('Resuming from checkpoint...')
        model_state_dict = ckpt['state_dict']
        for key in list(model_state_dict.keys()):
            if 'module.' in key:
                model_state_dict[key.replace('module.', '')] = model_state_dict.pop(key)
        model.load_state_dict(model_state_dict)

        best_cd_val = ckpt['best_cd_val']
        start_iter = ckpt['it']
    
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    if compress:
        criterion = DistortionVQLoss(alpha = args.vq_alpha)
    else:
        criterion = cal_loss
    # Main loop
    print('Start training...')
    try:
        it = start_iter

        while it <= args.max_iters:
            metric_logger = train(
                it,
                model,
                train_iterator,
                optimizer,
                scheduler,
                criterion,
                args,
                device,
                compress,
                pyg_ds
                )
            
            if log_wandb:
                if compress:
                    wandb.log({
                        f'train/distortion_vq_loss': metric_logger.distortion_vq_loss.global_avg,
                        f'train/distortion_loss': metric_logger.distortion_loss.global_avg,
                        f'train/vq_loss': metric_logger.vq_loss.global_avg,
                        f'train/n_points': metric_logger.n_points.global_avg,
                        f'lr': metric_logger.lr.global_avg
                    }, step = it)
                else:
                    wandb.log({
                        f'train/loss': metric_logger.loss.global_avg,
                        f'lr': metric_logger.lr.global_avg
                    }, step = it)
            
            
            
            if (it-1) % args.val_freq == 0 or it == args.max_iters:
                # with torch.no_grad():
                cd_val, md_val, n_points = validate_loss(it, model, val_loader, args,device, tag = 'Val', pyg_ds = pyg_ds)

                if log_wandb:
                    wandb.log({
                        f'val/cd': cd_val,
                        f'val/emd': md_val,
                        f'val/n_points': n_points
                    }, step = it)


                if True:
                    recons, origins = validate_inspect(it, model, val_loader, args,device, tag = 'Val', pyg_ds = pyg_ds)

                    for i in range(len(recons)):

                        if log_wandb:
                            wandb.log({
                                f"val/rec_{i}": wandb.Object3D({
                                    'type': 'lidar/beta',
                                    'points': recons[i][0].cpu().numpy()
                                }),
                                f"val/orig_{i}": wandb.Object3D({
                                    'type': 'lidar/beta',
                                    'points': origins[i][0].cpu().numpy()
                                })
                            }, step = it)
                
                    is_best = cd_val < best_cd_val
                    best_cd_val = min(cd_val, best_cd_val)
                    
                    states = {
                        'args': args,
                        'state_dict': model.state_dict(),
                        'state_dict_with_ddp': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_cd_val': best_cd_val,
                        'it': it
                    }
                    path = ckpt_mgr.save(states, is_best)
                    print(f'Saving checkpoints in: {path}')

            it += 1

    except KeyboardInterrupt:

        if log_wandb:
            wandb.run.finish()
        print('Terminating...')
    if log_wandb:
        wandb.run.finish()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()