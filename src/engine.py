import torch
from evaluation import EMD_CD, EMD_CD_non_batch


from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import time
import utils_ddp
import sys

# Train

def train(it, model, train_iterator, optimizer, scheduler, criterion, args, device, compress = False, pyg_ds = False):   
    optimizer.zero_grad()

    metric_logger = utils_ddp.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils_ddp.SmoothedValue(window_size=1, fmt="{value}"))

    model.train()

    # Load data
    batch = next(train_iterator)

    if not pyg_ds:
        x = batch['pointcloud'].to(device)


        B,N,C = x.shape
        x = x.reshape(-1,C) # B*N,C
        idxs = torch.arange(B)
        idxs = idxs.repeat_interleave(N).to(x.device) # (B*N)
    else:
        x = batch.pos.to(device)  # (B*N,3)
        idxs = batch.batch.to(device)  # (B*N)


    # Forward
    start_time = time.time()
    # loss = model.get_loss(x)
    
    pred, gold, qv_loss = model(x, batch_idxs = idxs)
    # loss, rec_loss = criterion(pred, gold, qv_loss)
    out_criterion = criterion(pred, gold, qv_loss)


    if compress:
        out_criterion['loss'].backward()
    else:
        out_criterion.backward()

    if args.max_grad_norm > 0:
        clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
    optimizer.step()
    scheduler.step()

    batch_time = time.time()-start_time 

    if compress:
        distortion_vq_loss = out_criterion["loss"].clone().detach()
        vq_loss = out_criterion["vq_loss"].clone().detach()
        distortion_loss = out_criterion["mse_loss"].clone().detach()

        if (it-1) % args.print_freq == 0 or it == args.max_iters:
            print('[Train] Iter %04d | DistVQLoss %.6f | BatchTime %.4f (s)' % 
                    (it, distortion_vq_loss, batch_time))
            

        metric_logger.update(
            distortion_vq_loss = distortion_vq_loss,
            vq_loss = vq_loss,
            distortion_loss = distortion_loss,
            lr=optimizer.param_groups[0]["lr"],
            n_points=0)
    else:
        if (it-1) % args.print_freq == 0 or it == args.max_iters:
            print('[Train] Iter %04d | Loss %.6f | BatchTime %.4f (s)' % 
                (it, out_criterion.item(), batch_time))
        metric_logger.update(loss=out_criterion.item(), lr=optimizer.param_groups[0]["lr"])

    return metric_logger    
    


def validate_loss(it, model, val_loader, args, device, tag = 'Val', pyg_ds = False):

    model.eval()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    metric_logger = utils_ddp.MetricLogger(delimiter="  ")

    # all_refs = []
    # all_recons = []

    num_processed_samples = 0

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(val_loader, desc=tag)):
            
            if args.num_val_batches > 0 and i >= args.num_val_batches:
                break
            
            if not pyg_ds:
                ref = batch['pointcloud'].to(device)
                shift = batch['shift'].to(device)
                scale = batch['scale'].to(device)
            else:
                ref = batch.pos.to(device)  # (N,3)
                shift = batch.shift.to(device)
                scale = batch.scale.to(device)
                ref = ref.unsqueeze(0)  # (1,N,3)

            batch_size = ref.shape[0]


            code = model.encode(ref)
            # code: [1, 256]
            recons = model.decode(code, ref.size(1), ref, flexibility=args.flexibility)
            # recons: [1, 2048, 3]
            
            ref = ref * scale + shift
            recons = recons * scale + shift

            metrics = EMD_CD_non_batch(recons, ref, eval_emd=False, verbose=False)
            # metrics = EMD_CD(recons, ref, batch_size=batch_size, eval_emd = False, verbose = False)

            cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()

            metric_logger.meters["cd"].update(cd, n=batch_size)
            metric_logger.meters["emd"].update(emd, n=batch_size)
            metric_logger.meters["n_points"].update(ref.size(1), n=batch_size)

            num_processed_samples += batch_size


    print('[Val] Iter %04d | CD %.6f | EMD %.6f  ' % 
          (it, metric_logger.cd.global_avg, metric_logger.emd.global_avg))

    return metric_logger.cd.global_avg, metric_logger.emd.global_avg, metric_logger.n_points.global_avg
    

def validate_inspect(it, model, val_loader, args, device, tag = 'Val', pyg_ds = False):
    model.eval()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    sum_n = 0

    all_x = []
    all_recons = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(val_loader, desc=f'Inspect {tag}')):
            if not pyg_ds:
                x = batch['pointcloud'].to(device)
            else:
                x = batch.pos.to(device)  # (N,3)
                x = x.unsqueeze(0)  # (1,N,3)
            
            code = model.encode(x)
            recons = model.decode(code, x.size(1), x, flexibility=args.flexibility).detach()

            all_x.append(x)
            all_recons.append(recons)

            sum_n += x.size(0)
            if i >= args.num_inspect_batches:
                break   # Inspect only 5 batch
    
    return all_recons, all_x
    
