import os
import time
import argparse
import torch
from tqdm.auto import tqdm

# from utils.dataset import *
# from utils.misc import *
# from utils.data import *
# from models.autoencoder import *
from evaluation import EMD_CD_non_batch, PSNR_non_batch, Density_non_batch
from utils.misc import seed_all
from utils.dataset_shapenet import ShapeNetCore
from utils.dataset_modelnet import ModelNet40

from torch.utils.data import DataLoader
from utils.misc import str_list
# from models.autoencoder import AutoEncoder
from models import AutoEncoder
from models import AutoEncoderVQ
import numpy as np

import sys
import glob
import open3d as o3d
import wandb
from statistics import mean
import json
import sys

# from denoising.utils.transforms import NormalizeUnitSphere
# from denoising.utils.denoise import patch_based_denoise
# from denoising.models import DenoiseNet


# FIRST time

# python -m evaluate_nonbatch.eval_model 
# --dataset shapenet 
# --ckpt results/shapenet/nocomp/ae_all_pointnet_bs128_different_CD_fz_diffusion_latent256_steps200_rotateFalse/model_best.pth.tar 
# --dataset-path /home/ids/gspadaro/repos/diffusion-point-cloud/data/shapenet.hdf5 
# --save-dir results/shapenet/nocomp/ae_all_pointnet_bs128_different_CD_fz_diffusion_latent256_steps200_rotateFalse/ 
# --save-recons
# --compressed-vq-model    

# If recons are already saved:
# python3 -m evaluate_nonbatch.eval_model 
# --save-dir results/shapenet/nocomp/ae_all_pointnet_bs128_different_CD_fz_diffusion_latent256_steps200_rotateFalse/ 
# --dataset shapenet 
# --compressed-vq-model    

def main(args, is_compression = False):

    
    
    os.makedirs(args.save_dir, exist_ok=True)

    # Checkpoint
    ckpt = torch.load(args.ckpt)
    seed_all(ckpt['args'].seed)

    
    

    # Datasets and loaders
    print('Loading datasets...')
    if args.dataset == 'shapenet':
        test_dset = ShapeNetCore(
            path=args.dataset_path,
            cates=['all'],
            split='test',
            scale_mode='shape_unit'
        )
        
    elif args.dataset == 'modelnet':
        test_dset = ModelNet40(
            path=args.dataset_path,
            num_points=2048,
            partition='test'
        )
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not yet implemented')
    
    bs = 128

    test_loader = DataLoader(test_dset, batch_size=bs, num_workers=0)

    # Model
    print('Loading model...')
    if not is_compression:
        model = AutoEncoder(ckpt['args']).to(args.device)
    else:
        model = AutoEncoderVQ(ckpt['args']).to(args.device) 

    model.load_state_dict(ckpt['state_dict'])
    model.eval()


    all_ref = []
    all_recons = []
    bpps = [] 
    for i, batch in enumerate(tqdm(test_loader)):
        ref = batch['pointcloud'].to(args.device)
        
        # print(ref.shape)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        with torch.no_grad():
            code = model.encode(ref)
            recons = model.decode(code, ref.size(1), ref, flexibility=ckpt['args'].flexibility).detach()
            
            if is_compression:
                num_points = ref.size(1)
                latent_dim = code.size(1)
                bpp = (8*(latent_dim//model.dim_codecs))/num_points  # 8 -> 7

                if i==0:
                    print(f'num_points: {num_points}')
                    print(f'latent_dim: {latent_dim}')
                    print(f'bpp: {bpp}')
                
                # bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_points
                bpps.append(bpp)       

        ref = ref * scale + shift
        recons = recons * scale + shift

        all_ref.append(ref.detach().cpu())
        all_recons.append(recons.detach().cpu())

    all_ref = torch.cat(all_ref, dim=0)
    all_recons = torch.cat(all_recons, dim=0)

    # print(torch.min(all_ref), torch.max(all_ref))
    # print(torch.min(all_recons), torch.max(all_recons))
    # print(f'BPP: {bpps}')
    if len(bpps) > 0:
        print(f'BPP avg: {mean(bpps)}') 
    print('Saving point clouds...')
    np.save(os.path.join(args.save_dir, 'ref.npy'), all_ref.numpy())
    np.save(os.path.join(args.save_dir, 'out.npy'), all_recons.numpy())
    np.save(os.path.join(args.save_dir, 'bpp.npy'), np.array(bpps))



    return all_ref, all_recons, torch.tensor(bpps, dtype = torch.float)

    

def compute_metrics(all_ref, all_recons, dataset):

    all_ref = all_ref #[:10] #*(10**3)
    all_recons = all_recons #[:10] #*(10**3)

    print('Start computing metrics...')
    metrics = EMD_CD_non_batch(all_recons, all_ref, reduced=True, eval_emd = True, eval_cd=True)
    cd = metrics['MMD-CD'].item()
    emd = metrics['MMD-EMD'].item()

    metrics = PSNR_non_batch(all_recons, all_ref, reduced=True, dataset = dataset)
    psnr = metrics['PSNR-D2'].item()

    metrics = Density_non_batch(all_recons, all_ref, reduced=True)
    density = metrics['Density'].item()
    
    return cd, emd, psnr, density



if __name__ == '__main__':

    
    # os.environ['PYTORCH_USE_CUDA_DSA'] = "1"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./pretrained/AE_airplane.pt')
    parser.add_argument('--categories', type=str_list, default=['all'])
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--dataset', type=str, default='shapenet')
    parser.add_argument('--dataset-path', type=str, default='./data/shapenet.hdf5')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-recons', action='store_true')
    parser.add_argument('--compressed-vq-model', action='store_true')

    



    args = parser.parse_args()


    if not args.compressed_vq_model:
        save_stats_results = f'{args.save_dir}/results_nocomp_model_emd.json'
        dim_codecs = None
    else:
        dim_codecs = str(args.save_dir).split('dim_codecs')[-1].replace('/','') 
        save_stats_results = f'{args.save_dir}/results_model_dim_codecs_{dim_codecs}_new_emd.json'
    


    log_wandb = False
    eval_metrics = True

    if log_wandb:
        wandb.login()

    
    

    if args.save_recons: 
        all_ref, all_recons, bpps = main(args, is_compression = args.compressed_vq_model)
    else:
        all_ref, all_recons, bpps = np.load(os.path.join(args.save_dir, 'ref.npy')), np.load(os.path.join(args.save_dir, 'out.npy')), np.load(os.path.join(args.save_dir, 'bpp.npy'))
        all_ref, all_recons, bpps = torch.from_numpy(all_ref), torch.from_numpy(all_recons), torch.from_numpy(bpps)

    

    # print('minmax ref')
    # print(torch.min(all_ref), torch.max(all_ref)) # [-1,1]
    # print('minmax recons')
    # print(torch.min(all_recons), torch.max(all_recons)) # [-1,1]
    # print(all_ref.shape) # torch.Size([7694, 2048, 3])


    cd, emd = 0.0, 0.0
    psnr = 0.0

    results = {}
    if eval_metrics:
        cd, emd, psnr, density = compute_metrics(all_ref, all_recons, args.dataset) 



        if not args.compressed_vq_model:
            print(f'(MODEL no comp) CD:  {cd}' )
            print(f'(MODEL no comp) EMD: {emd}')
            print(f'(MODEL no comp) PSNR: {psnr}')
            print(f'(MODEL no comp) Desnity: {density}')
            results['no_comp'] = {
                'cd': cd,
                'emd': emd,
                'psnr': psnr,
                'density': density,
            }
        else:
            print(f'(MODEL dim_codecs= {dim_codecs}) BPP:  {torch.mean(bpps).item()}' )
            print(f'(MODEL dim_codecs= {dim_codecs}) CD:  {cd}' )
            print(f'(MODEL dim_codecs= {dim_codecs}) EMD: {emd}')
            print(f'(MODEL dim_codecs= {dim_codecs}) PSNR: {psnr}')
            print(f'(MODEL dim_codecs= {dim_codecs}) Density: {density}')
            results[f'dim_codecs_{dim_codecs}'] = {
                'bpp': torch.mean(bpps).item(),
                'cd': cd,
                'emd': emd,
                'psnr': psnr,
                'density': density,
            }


    
    with open(save_stats_results, "w") as file:
            json.dump(results, file, indent=4) 


    print('----------------\n\n')

    if log_wandb:
        if not args.compressed_vq_model:
            wandb.init(
                # Set the project where this run will be logged
                project='PointCloud_Compression_Eval', 
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name = f'pointnet_compression_model_no_comp',
                config = vars(args))

            wandb.log({
                    f'metrics/cd': cd,
                    f'metrics/emd': emd,
                }, step = 0)
            for i in range(2):
                wandb.log({
                    f"PointClouds_no_comp/rec_{i}": wandb.Object3D({
                        'type': 'lidar/beta',
                        'points': all_recons[i].cpu().numpy()
                    }),
                    f"PointClouds_no_comp/orig_{i}": wandb.Object3D({
                        'type': 'lidar/beta',
                        'points': all_ref[i].cpu().numpy()
                    })
                }, step = 0)
        else:
            wandb.init(
                # Set the project where this run will be logged
                project='PointCloud_Compression_Eval', 
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name = f'pointnet_compression_model_dim_codecs_{dim_codecs}',
                config = vars(args))

            wandb.log({
                    'bpp': torch.mean(bpps).item(),
                    f'metrics/cd': cd,
                    f'metrics/emd': emd,
                }, step = 0)
            for i in range(2):
                wandb.log({
                    f"PointClouds_dim_codecs_{dim_codecs}/rec_{i}": wandb.Object3D({
                        'type': 'lidar/beta',
                        'points': all_recons[i].cpu().numpy()
                    }),
                    f"PointClouds_dim_codecs_{dim_codecs}/orig_{i}": wandb.Object3D({
                        'type': 'lidar/beta',
                        'points': all_ref[i].cpu().numpy()
                    })
                }, step = 0)

    