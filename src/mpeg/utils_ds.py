from utils.dataset_shapenet import ShapeNetCore
from utils.dataset_modelnet import ModelNet40
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def get_references(dataset = 'shapenet', dataset_path = None, device = 'cuda'):
    
    # Datasets and loaders
    print('Loading datasets...')

    if dataset == 'shapenet':
        test_dset = ShapeNetCore(
            path=dataset_path,
            cates=['all'],
            split='test',
            scale_mode='shape_unit'
        )
        
    elif dataset == 'modelnet':
        test_dset = ModelNet40(
            path=dataset_path,
            num_points=2048,
            partition='test'
        )
    else:
        raise NotImplementedError(f'Dataset {dataset} not yet implemented')

    test_loader = DataLoader(test_dset, batch_size=128, num_workers=0)

    all_ref = []

    for i, batch in enumerate(tqdm(test_loader)):
        ref = batch['pointcloud'].to(device)
        
        shift = batch['shift'].to(device)
        scale = batch['scale'].to(device)

        
        ref = ref * scale + shift

        all_ref.append(ref.detach().cpu())

    all_ref = torch.cat(all_ref, dim=0)
    # TODO don't cat them 

    # np.save(os.path.join(args.save_dir, 'ref.npy'), all_ref.numpy())



    return all_ref