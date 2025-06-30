from utils.misc import THOUSAND, str_list
import argparse


def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1

def get_args():
    # Arguments
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num-steps', type=int, default=200)
    parser.add_argument('--beta-1', type=float, default=1e-4)
    parser.add_argument('--beta-T', type=float, default=0.05)
    parser.add_argument('--sched-mode', type=str, default='linear')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--resume', type=str, default=None)

    # Datasets and loaders
    parser.add_argument('--dataset-path', type=str, default='./data/shapenet.hdf5')
    parser.add_argument('--categories', type=str_list, default=['all'])
    parser.add_argument('--scale-mode', type=str, default='shape_unit')
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--val-batch-size', type=int, default=32)
    parser.add_argument('--rotate', type=int2bool, default=0) # 1 == True

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10)
    parser.add_argument('--end-lr', type=float, default=1e-4)
    parser.add_argument('--sched-start-epoch', type=int, default=200000)
    parser.add_argument('--sched-end-epoch', type=int, default=400000)

    # Training
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log-root', type=str, default='./logs_ae')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-iters', type=int, default=1000000)
    parser.add_argument('--val-freq', type=float, default=10000.0)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--num-val-batches', type=int, default=30)
    parser.add_argument('--num-inspect-batches', type=int, default=1)
    parser.add_argument('--num-inspect-pointclouds', type=int, default=4)


    # ours
    parser.add_argument('--save-dir', type=str, default='./exp')
    parser.add_argument('--test-freq', type=float, default=10000.0)
    parser.add_argument('--print-freq', type=float, default=100)
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet','modelnet'])
    parser.add_argument('--encoder', type=str, default='pointnet') #, choices=['pointnet','gcn', 'vig','pointnet2','pointnet2_orig','dgcnn'])

    # Pointnet++
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--use-dropout', type=int2bool, default=1) # 1 == True



    # DeepGCN
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--n-blocks', type=int, default=7)
    parser.add_argument('--block', type=str, default='res')

    # DGCNN
    parser.add_argument('--emb-dims', type=int, default=1024)
    
    parser.add_argument('--dropout', type=float, default=0.0)



    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")



    parser.add_argument("--local-rank","--local_rank", default=0, type=int)
    parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
    parser.add_argument("--init-method", default='env://', type=str)
    



    # Vector Quantization
    parser.add_argument('--num-codecs', type=int, default=128)
    parser.add_argument('--dim-codecs', type=int, default=16)
    parser.add_argument('--vq-alpha', type=float, default=10.0)



    parser.add_argument('--train_data_path')
    parser.add_argument('--val_data_path')
    parser.add_argument('--use_instance_norm', type=int2bool, default=0) # 1 == True
    parser.add_argument('--std-noise-diffusion', type=float, default=-1)


    args = parser.parse_args()
    return args