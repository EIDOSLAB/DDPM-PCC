from models import AutoEncoder
from vector_quantize_pytorch import VectorQuantize
import torch
import sys

import time
import statistics

import sys

class AutoEncoderVQ(AutoEncoder):
    def __init__(self, args):
        super().__init__(args)

        # latent -> [----] 256 -> [--] [--] .. [--] 16
        # qt -> idx: 0..511
        # diff(idx)
        assert args.latent_dim % args.dim_codecs == 0, 'Codbooks vector dimensions have to be a multiple of the latent dimension.'
        
        print(f'AutoEncoderVQ:\ndim_codecs: {args.dim_codecs}\nnum_codecs: {args.num_codecs}')
        
        self.dim_codecs = args.dim_codecs
        self.vq = VectorQuantize(
            dim = args.dim_codecs,
            codebook_size = args.num_codecs 
        )

    def quantize_single_vector(self, x):
        # x: B, latent_dim
        B, latent_dim = x.shape
        x = x.reshape(B, latent_dim//self.dim_codecs, self.dim_codecs)
        quantized, indices, commit_loss = self.vq(x)
        # print(f'idx: {indices.shape}'): B, latent_dim//self.dim_codecs
        # print(quantized.shape): B, latent_dim//self.dim_codecs, self.dim_codecs
        quantized = quantized.reshape(B, -1)

        return quantized, indices, commit_loss

    def encode(self, x, batch_idxs = None): 
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x, batch_idxs = batch_idxs)
        quantized, indices, commit_loss = self.quantize_single_vector(code)
        return quantized

    def decode(self, code, num_points, x_0, flexibility=0.0, ret_traj=False): 

        return self.diffusion.sample(num_points, code, x_0, flexibility=flexibility, ret_traj=ret_traj)
    
    def forward(self, x, batch_idxs = None): 
        # x: (B, N, 3) or (B*N, 3)
        code, _ = self.encoder(x, batch_idxs = batch_idxs)
        # code: B, latent_dim
        quantized, indices, commit_loss = self.quantize_single_vector(code)
        # quantized: B, latent_dim

        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        e_theta, e_rand = self.diffusion.get_elements(x, quantized, batch_idxs = batch_idxs)

        # e_theta, e_rand: B*N, 3

        
        # loss = F.mse_loss(e_theta, e_rand, reduction='mean')
        # print(loss)
        return e_theta, e_rand, commit_loss


