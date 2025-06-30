import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
import glob
import sys
import torch

from mpeg.parallel_process import parallel_process, Popen

from .utils_ds import get_references

# from src folder:
# python -m mpeg.run

# ENCODER / DECODER
# /home/ids/gspadaro/repos/mpeg-pcc-tmc13/build/tmc3/tmc3 
# -c ../assets/octree/r04/encoder.cfg 
# --uncompressedDataPath=test_ascii.ply 
# --compressedStreamPath=test_ascii_compr.bin 
# --reconstructedDataPath=test_ascii_rec.ply

# VERIFY
# >>> import numpy as np
# >>> import open3d as o3d
# >>> np.asarray(o3d.io.read_point_cloud("test_ascii_rec.ply").points)
# python -c "import numpy as np; import open3d as o3d; print(np.asarray(o3d.io.read_point_cloud('test_ascii_rec.ply').points).shape)"


MPEG_TMC13_DIR = '/home/ids/gspadaro/repos/mpeg-pcc-tmc13/'
TMC13 = f'{MPEG_TMC13_DIR}/build/tmc3/tmc3'

def normalize_pointclouds(pc, max_value = 2048, get_torch = True):
    translated = pc + 1
    # print(torch.min(translated), torch.max(translated)) # [0,2]

    scaled = (translated / 2) * max_value
    # print(torch.min(scaled), torch.max(scaled)) # [0,MAX]

    if get_torch:
        norm = scaled.to(torch.float32)
    else:
        norm = scaled.astype(np.float32)

    return norm


def unnormalize_pointclouds(pc, max_value = 2048, get_torch = True):

    pc = (pc / max_value) * 2
    scaled = pc - 1

    # translated = pc + 1
    # # print(torch.min(translated), torch.max(translated)) # [0,2]

    # scaled = (translated / 2) * max_value
    # # print(torch.min(scaled), torch.max(scaled)) # [0,MAX]

    if get_torch:
        norm = scaled.to(torch.float32)
    else:
        norm = scaled.astype(np.float32)

    return norm


def run_mpeg_exp(mpeg_cfg_path, current_mpeg_output_dir, idx, pc):
    # command = f'{TMC13} '
    # command +=f'-c {mpeg_cfg_path}/encoder.cfg '
    # command +=f'--uncompressedDataPath={pc} '
    # command +=f'--compressedStreamPath={current_mpeg_output_dir}/compressed/ref_{idx}.bin '
    # command +=f'--reconstructedDataPath={current_mpeg_output_dir}/reconstructed/ref_{idx}.bin '
    # os.system(command)
    # sys.exit(1)

    out = open(f'{current_mpeg_output_dir}/encodings_{idx}.out', "a")
    err = open(f'{current_mpeg_output_dir}/encodings_{idx}.err', "a")

    return Popen([
        f'{TMC13}',
        '-c', f'{mpeg_cfg_path}/encoder.cfg',
        f'--uncompressedDataPath={pc}',
        f'--compressedStreamPath={current_mpeg_output_dir}/compressed/ref_{idx}.bin',
        f'--reconstructedDataPath={current_mpeg_output_dir}/reconstructed/ref_{idx}.ply'
    ], 
    stdout=out, 
    stderr=err)


if __name__ == '__main__':

    create_ply_ref = True
   
    ref_path = None

    dataset = 'shapenet'
    dataset_path = '../datasets/data/shapenet.hdf5'
    saved_exp_dir = '../results/shapenet/'
    # dataset = 'modelnet'
    # dataset_path = '../datasets/'
    # saved_exp_dir = '../results/modelnet/'
    
    configs_path = '../assets/'

    if ref_path is None:
        print('Generating references in .npy format..')
        all_ref = get_references(dataset=dataset, dataset_path=dataset_path, device='cuda')
        ref_path = os.path.join(saved_exp_dir, 'ref.npy')
        np.save(ref_path, all_ref.numpy())

    
    
    if create_ply_ref:
        print(f'Creating .ply files from {ref_path}')
        all_ref = np.load(ref_path)

        os.makedirs(f'{saved_exp_dir}/ref_ply',exist_ok=True)
        print(f'Results will be saved in {saved_exp_dir}/ref_ply')
        print('[ATTENTION] points coordinates are normalized between 0 and 128 before saving it !!')

        for i,xyz in tqdm(enumerate(all_ref)):
            dtype = o3d.core.float32
            xyz = normalize_pointclouds(xyz, max_value=128, get_torch=False)
            # print(np.min(xyz), np.max(xyz))
            # sys.exit(1)

            p_tensor = o3d.core.Tensor(xyz, dtype=dtype)
            pc = o3d.t.geometry.PointCloud(p_tensor)
            o3d.t.io.write_point_cloud(f'{saved_exp_dir}/ref_ply/ref_ascii_{i}.ply', pc, write_ascii = True)


    list_of_points = glob.glob(f'{saved_exp_dir}/ref_ply/ref_ascii_*.ply')
    assert len(list_of_points) > 0, f'Before running the encodings you have to convert the .npy file into .ply pointclouds'
    print(f'Founded {len(list_of_points)} pointclouds!')


    output_encodings_path = f'{saved_exp_dir}/mpeg_small/'
    os.makedirs(output_encodings_path,exist_ok=True)
    print(f'Starting Encodings, results will be saved in {output_encodings_path}')
    

    mpeg_modes = {
        # 'same_rate_merge_modelnet': ['r01', 'r02', 'r03', 'r04', 'r05', 'r06']
        'same_rate_merge_shapenet': ['r01', 'r02', 'r03', 'r04', 'r05', 'r06']

    }

    params = []
    for mode in mpeg_modes.keys():
        print(f'Evaluating mode {mode}')
        mpeg_output_dir = os.path.join(output_encodings_path, mode)
        os.makedirs(mpeg_output_dir,exist_ok=True)
        
        for rate in mpeg_modes[mode]:
            print(f'[{mode}] ({rate}):')
            current_mpeg_output_dir = os.path.join(mpeg_output_dir, rate)
            os.makedirs(current_mpeg_output_dir, exist_ok=True)
            os.makedirs(f'{current_mpeg_output_dir}/compressed', exist_ok=True)
            os.makedirs(f'{current_mpeg_output_dir}/reconstructed', exist_ok=True)

            mpeg_cfg_path = f'{configs_path}/{mode}/{rate}/'
            
            for idx in tqdm(range(len(list_of_points))):

                pc = f'{saved_exp_dir}/ref_ply/ref_ascii_{idx}.ply'
                # ENCODER / DECODER
                run_mpeg_exp(mpeg_cfg_path, current_mpeg_output_dir, idx, pc)
                
                # params.append((mpeg_cfg_path, current_mpeg_output_dir, idx, pc))
                # break
            # break
        # break
    
    # print('Started GPCC experiments')
    # # An SSD is highly recommended, extremely slow when running in parallel on an HDD due to parallel writes
    # # If HDD, set parallelism to 1
    # parallel_process(run_mpeg_exp, params, 1)
    # print('Finished GPCC experiments')

                

            