import os
import argparse
import torch

# from utils.dataset import *
# from utils.misc import *
# from utils.data import *
# from models.autoencoder import *
from evaluation import EMD_CD_non_batch, PSNR_non_batch, Density_non_batch
import numpy as np

import sys
from mpeg.run import normalize_pointclouds, unnormalize_pointclouds
from mpeg.find_bpp import get_bpp, get_bpp_with_memory, get_bpp_with_memory_draco
import glob
import open3d as o3d
import wandb
import json
import pickle


# python -m evaluate_nonbatch.eval_gpcc

def compute_metrics(all_ref, all_recons, dataset):

    all_ref = all_ref #[:10] #*(10**3)
    all_recons = all_recons #[:10] #*(10**3)

    print('Start computing metrics...')
    metrics = EMD_CD_non_batch(all_recons, all_ref, reduced=False, eval_emd = True, eval_cd=True)
    cd = metrics['MMD-CD'].item()
    emd = metrics['MMD-EMD'].item()

    metrics = PSNR_non_batch(all_recons, all_ref, reduced=False, dataset = dataset)
    psnr = metrics['PSNR-D2'].item()
    # psnr = 0

    metrics = Density_non_batch(all_recons, all_ref, reduced=False)
    density = metrics['Density'].item()
    # density = 0
    
    return cd, emd, psnr, density



if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--mpeg-encoded-path', type=str, default=None)
    parser.add_argument('--mpeg-ref-ply', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='modelnet')

    # parser.add_argument('--batch_size', type=int, default=128)



    args = parser.parse_args()

    log_wandb = False
    eval_metrics = True

    if log_wandb:
        wandb.login()



    # MPEG
    if args.dataset == 'modelnet':
        args.mpeg_encoded_path = '../results/modelnet/mpeg/same_rate_merge_modelnet'
        args.mpeg_ref_ply = '../results/modelnet/ref_ply/'
        save_stats_results = f'{args.mpeg_encoded_path}/results_gpcc_new_emd.json'
        dataset = 'modelnet'
    else:
        args.mpeg_encoded_path = '../results/shapenet/mpeg/same_rate_merge_shapenet'
        args.mpeg_ref_ply = '../results/shapenet/ref_ply/'
        save_stats_results = f'{args.mpeg_encoded_path}/results_gpcc_new_emd.json'
        dataset = 'shapenet'





    configs = args.mpeg_encoded_path.split('mpeg/')[-1].replace('/','')  # dummy_same_rate_merge (mpeg configuration)



    results = {}
    if args.mpeg_encoded_path is not None:

        origing_path_mpeg = f'{args.mpeg_ref_ply}/ref_ascii_*.ply'
        num_origins_mpeg = len(glob.glob(origing_path_mpeg))
        print(f'Founded {num_origins_mpeg} references pointclouds (.ply)')

        all_ref_mpeg = []
        for idx in range(num_origins_mpeg):
            ref = np.asarray(o3d.io.read_point_cloud(origing_path_mpeg.replace('*',str(idx))).points)
            ref = torch.from_numpy(ref).unsqueeze(0)
            # TODO normalize rec between -1...1
            ref = unnormalize_pointclouds(ref, max_value= 128, get_torch=True)
            all_ref_mpeg.append(ref)

        

        for rate in range(len(os.listdir(args.mpeg_encoded_path))):
            
            rate = f'r0{rate+1}'

            if not os.path.isdir(os.path.join(args.mpeg_encoded_path,rate)):
                continue

            # if not rate.startswith('r0'):
            # continue
            
            reconstructed_paths = os.path.join(args.mpeg_encoded_path,rate, 'reconstructed')
            # compressed_paths = os.path.join(args.mpeg_encoded_path,rate, 'compressed')
            encoding_out_file = os.path.join(args.mpeg_encoded_path,rate, 'encodings.out')

            mpeg_recons_path = f'{reconstructed_paths}/ref_*.ply'
            num_mpeg_recons = len(glob.glob(mpeg_recons_path))
            print(f'Founded {num_mpeg_recons} reconstructed pointclouds (rate {rate})')

            all_rec_mpeg = []
            for idx in range(num_mpeg_recons):
                rec = np.asarray(o3d.io.read_point_cloud(mpeg_recons_path.replace('*',str(idx))).points)
                rec = torch.from_numpy(rec).unsqueeze(0)
                # TODO normalize rec between -1...1
                rec = unnormalize_pointclouds(rec, max_value= 128, get_torch=True)

                all_rec_mpeg.append(rec)

            # with open(os.path.join(args.mpeg_encoded_path, f'rec_{rate}.npy'), 'wb') as file:
            #     pickle.dump(all_rec_mpeg, file)

            # with open(os.path.join(args.mpeg_encoded_path, f'ref_{rate}.npy'), 'wb') as file:
            #     pickle.dump(all_ref_mpeg, file)

                




            bpp = get_bpp_with_memory(encoding_out_file, n_points=2048)

            # print('minmax ref (after norm)')
            # print(torch.min(all_ref_mpeg), torch.max(all_ref_mpeg)) # [0,MAX]
            # print('minmax recons (after norm)')
            # print(torch.min(all_rec_mpeg), torch.max(all_rec_mpeg)) # [0,MAX]


            cd, emd = 0.0, 0.0
            psnr, density = 0.0, 0.0
            if eval_metrics:
                cd, emd, psnr, density = compute_metrics(all_ref_mpeg, all_rec_mpeg, dataset) 

                

                print(cd)
                print(emd)
                cd, emd = cd.mean(), emd.mean()
                print(f'({rate}) BPP:  {bpp}')
                print(f'({rate}) CD:  {cd}')
                print(f'({rate}) EMD:  {emd}')
                print(f'({rate}) PSNR:  {psnr}')
                print(f'({rate}) DENSITY:  {density}')

                results[rate] = {
                    'bpp':bpp,
                    'cd': cd,
                    'emd': emd,
                    'psnr': psnr,
                    'density': density
                }


        
        with open(save_stats_results, "w") as file:
            json.dump(results, file, indent=4) 

            