from torch.nn import Module

from .encoders import *
from .diffusion import *

import time
import statistics

class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.encoder == 'pointnet':
            print('USING POINTNET ENCODER')
            use_instance_norm = False
            is_legacy_encoder = True
            if hasattr(args, "use_instance_norm"):
                use_instance_norm = args.use_instance_norm
                is_legacy_encoder = False
            
            if is_legacy_encoder:
                print('\n\n!!!! [ATTENTION] Loading a Legacy model !!!!\n\n')

            self.encoder = PointNetEncoder(zdim=args.latent_dim, use_instance_norm = use_instance_norm, legacy=is_legacy_encoder)

        elif args.encoder == 'pointnet2':
            print('USING POINTNET++ ENCODER')
            self.encoder = PointNet2Encoder(z_dim=args.latent_dim)

        elif args.encoder == 'pointnet2_orig':
            print(f'USING POINTNET++ (orig. mode {args.mode}) ENCODER (dropout: {args.use_dropout})')
            self.encoder = PointNetOrig2Encoder(z_dim=args.latent_dim, mode=args.mode, use_dropout=args.use_dropout)



        print('Using diffusion model!!')
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x, batch_idxs = None):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x, batch_idxs = batch_idxs)
        return code

    def decode(self, code, num_points, x_0, flexibility=0.0, ret_traj=False):

        return self.diffusion.sample(num_points, code, x_0, flexibility=flexibility, ret_traj=ret_traj)
    

    def forward(self, x, batch_idxs = None): 
        # x: (B, N, 3) or (B*N, 3)
        code, _ = self.encoder(x, batch_idxs = batch_idxs)
        # code: B, latent_dim

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # x: [B*N,1,3]       
        # code: [B, 256]         
        # batch_idxs: [B*N,]       
        e_theta, e_rand = self.diffusion.get_elements(x, code, batch_idxs = batch_idxs)
        # e_theta, e_rand: B*N, 3

        return e_theta, e_rand, None
