import math
import torch
from torch import nn
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))



class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        diffusion_net,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.diffusion_net = diffusion_net
        self.loss_type = loss_type
        self.cond = conditional
        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        print(self.loss_type)
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        gamma = np.cumprod(alphas, axis=0)
        gamma_tmin1 = np.append(1., gamma[:-1])
        self.sqrt_gamma_tmin1 = np.sqrt(
            np.append(1., gamma))

        timesteps, = betas.shape
        self.timesteps_n = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('gamma', to_torch(gamma))
        self.register_buffer('gamma_tmin1',to_torch(gamma_tmin1))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_frac_gamma',to_torch(np.sqrt(1. / gamma)))
        self.register_buffer('sqrt_frac_gamma_min1',to_torch(np.sqrt(1. / gamma - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        #######################
        ###       TODO      ###
        #######################
        posterior_var = betas * (1. - gamma_tmin1) / (1. - gamma)

        self.register_buffer('posterior_var',to_torch(posterior_var))
        self.register_buffer('log_posterior_var', to_torch(np.log(np.maximum(posterior_var, 1e-20))))
        self.register_buffer('posterior_mean1', to_torch(betas * np.sqrt(gamma_tmin1) / (1. - gamma)))
        self.register_buffer('posterior_mean2', to_torch((1. - gamma_tmin1) * np.sqrt(alphas) / (1. - gamma)))


    def predict_start_from_noise(self, x_t, t, noise):
        ''' Hint: variable at "t" (use like "some_variable[t]") '''
        return (x_t - self.sqrt_frac_gamma_min1[t] * noise) / self.sqrt_frac_gamma[t]
        #######################
        ###       TODO      ###
        #######################

    def q_posterior(self, x_first, x_t, t):
        # mean of q(x_{t-1} | x_t, x_0)
        posterior_mean = self.posterior_mean1[t] * x_first + self.posterior_mean2[t] * x_t
        #######################
        ###       TODO      ###
        #######################
        log_posterior_var = self.log_posterior_var[t]
        return posterior_mean, log_posterior_var

 
    def p_mean_variance(self, x, t, clip_denoised: bool, noisy_img=None): 
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_gamma_tmin1[t+1]]).repeat(batch_size, 1).to(x.device)

        if noisy_img is not None:
            x = torch.cat([noisy_img, x], dim=1)
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.diffusion_net(x, noise_level))
            x_recon = x_recon[:, self.channels:, :, :]
            #######################
            ###       TODO      ###
            #######################
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.diffusion_net(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, log_posterior_var = self.q_posterior(
            x_first=x_recon, x_t=x, t=t)
        return model_mean, log_posterior_var

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, noisy_img=None):
        model_mean, log_model_var = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, noisy_img=noisy_img)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * log_model_var).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, noisy_img, continous=False):
        device = self.betas.device

        x = noisy_img
        shape = x.shape
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.timesteps_n)), desc='sampling time step', total=self.timesteps_n):
            img = self.p_sample(img, i, noisy_img=x)
        if continous:
            return img
        else:
            return img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def sampling(self, x_in, noisy_img, continous=False):
        return self.p_sample_loop(x_in, noisy_img, continous) 

    def q_sample(self, x_first, continuous_sqrt_gamma, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_first))

        return (
            continuous_sqrt_gamma * x_first +
            (1 - continuous_sqrt_gamma**2).sqrt() * noise
        )

    def p_losses(self, x_in, noisy_img, noise=None):
        x_first = x_in
        [b, c, h, w] = x_first.shape
        t = np.random.randint(1, self.timesteps_n + 1)

        continuous_sqrt_gamma = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_gamma_tmin1[t-1],
                self.sqrt_gamma_tmin1[t],
                size=b
            )
        ).to(x_first.device)

        continuous_sqrt_gamma = continuous_sqrt_gamma.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_first))
        x_noisy = self.q_sample(x_first=x_first, 
                                continuous_sqrt_gamma=continuous_sqrt_gamma.view(-1, 1, 1, 1), 
                                noise=noise)

        if not self.cond: 
            x_recon = self.diffusion_net(x_noisy, continuous_sqrt_gamma)
        else: # compute loss function 
            #######################
            ###       TODO      ###
            #######################
            x_recon = self.diffusion_net(torch.cat([noisy_img, x_noisy], dim=1), continuous_sqrt_gamma)
        
        loss = self.loss_func(noise, x_recon)
   
        return loss

    def forward(self, x_in, noisy_img, *args, **kwargs):
        return self.p_losses(x_in, noisy_img, *args, **kwargs) 

