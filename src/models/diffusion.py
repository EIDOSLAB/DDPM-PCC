import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
from torch.nn import Linear
import random
import sys
# from .common import *

class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        # print(x.shape)
        # print(ctx.shape)

        # x:    (B, N, dim_in)
        # ctx:  (B, 1, dim_ctx)

        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret

class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 3, context_dim+3)
        ])

    def forward(self, x, beta, context, idxs = None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = beta.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        if idxs is not None:
            ctx_emb = ctx_emb[idxs] # (B*N,1,F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out



class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule,  std_noise = None):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        print(f'Real Diffusion')

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        x_t = c0 * x_0 + c1 * e_rand
        e_theta = self.net(x_t, beta=beta, context=context)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss
    
    def get_elements(self, x_0, context, t=None, batch_idxs = None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            or
            x_0:  Input point cloud, (B*N, 1, d).

            context:  Shape latent, (B, F).
        """
        if batch_idxs is None:
            batch_size, _, point_dim = x_0.size()
        else:
            _, _, point_dim = x_0.size()
            batch_size = torch.max(batch_idxs).item()+1
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size) # [t1, t2, ...] len = B

        alpha_bar = self.var_sched.alpha_bars[t]    # (B)
        beta = self.var_sched.betas[t]              # (B)

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        if batch_idxs is not None:

            c0 = c0[batch_idxs]  # (B*N, 1, 1)
            c1 = c1[batch_idxs]  # (B*N, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d) or (B*N, 1, d) 
        x_t = c0 * x_0 + c1 * e_rand    # (B, N, d) or (B*N, 1, d)

        e_theta = self.net(x_t, beta=beta, context=context, idxs = batch_idxs) # (B, N, d) or (B*N, 1, d)
        
        return e_theta.view(-1, point_dim), e_rand.view(-1, point_dim) 
    

    def sample(self, num_points, context, x_0, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]






if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    batch_size = 8
    code = torch.rand((batch_size, 256))
    x = torch.rand((batch_size,5,3))

    x = x.view(-1,1,3)

    idxs = torch.arange(batch_size)
    idxs = idxs.repeat_interleave(5)

    diffusion = DiffusionPoint(
        net = PointwiseNet(point_dim=3, context_dim=256, residual=True),
        var_sched = VarianceSchedule(
            num_steps=200,
            beta_1=0.0001,
            beta_T=0.05,
            mode='linear'
        )
    )


    e_theta, e_rand = diffusion.get_elements(x, code, batch_idxs = idxs)
    
    # torch.save(e_theta,'e_theta_flat_check.pth')
