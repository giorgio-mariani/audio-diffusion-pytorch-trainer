from pathlib import Path
from typing import List, Optional, Callable, Tuple

import numpy as np
import torch
import torchaudio
import tqdm
from torch import Tensor
from math import sqrt

from audio_diffusion_pytorch.diffusion import Sampler, KarrasSchedule, Schedule
from torch.utils.data import DataLoader

from main.dataset import assert_is_audio, SeparationDataset
from main.separation import least_squares_normalization


class EulerExtractor(Sampler):

    def __init__(self, mixture: torch.Tensor):
        super().__init__()
        self.mixture = mixture

    def compute_differential(self, s1, s2, x, sigma):
        grad_log_p1 = s1(x, sigma) - 1. * s2(self.mixture - x, sigma).mean(dim=0).unsqueeze(0)
        return -grad_log_p1

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        # Sigma steps
        S1 = lambda x, sigma: (fns[0](x, sigma=sigma) - x)/sigma
        S2 = lambda x, sigma: (fns[1](x, sigma=sigma) - x)/sigma

        xs_next = []
        for x, (s1,s2) in zip(xs, [(S1, S2), (S2, S1)]):
            # Derivative at sigma (∂x/∂sigma)
            d = self.compute_differential(s1=s1, s2=s2, x=x, sigma=sigma)

            # Euler step
            x_next = x + d * (sigma_next - sigma)

            xs_next.append(x_next)
        return xs_next

    def forward(
        self, noises: Tensor, fns: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        xs = [sigmas[0] * noise for noise in noises]
        # Denoise to sample
        for i in range(num_steps - 1):
            xs = self.step(xs, fns=fns, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa

        y1, y2 = xs
        y1, y2 = self.mixture - y2, self.mixture - y1

        y1_normalized, y2_normalized = least_squares_normalization(y1, y2, self.mixture)
        #y1_cons, y2_cons = enforce_mixture_consistency(self.mixture, torch.stack([y1_normalized, y2_normalized], dim=1))

        return [y1_normalized, y2_normalized]
    

class RungeKuttaExtractor(Sampler):

    def __init__(self, mixture: torch.Tensor):
        super().__init__()
        self.mixture = mixture

    def compute_differential(self, s1, s2, x, sigma):
        grad_log_p1 = s1(x, sigma) - 1. * s2(self.mixture - x, sigma).mean(dim=0).unsqueeze(0)
        return -grad_log_p1

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        # Positive gradients
        S1 = lambda x, sigma: (fns[0](x, sigma=sigma) - x)/sigma
        S2 = lambda x, sigma: (fns[1](x, sigma=sigma) - x)/sigma

        xs_next = []
        for x, (s1,s2) in zip(xs, [(S1, S2), (S2, S1)]):
            h = sigma_next - sigma #negative
            f = self.compute_differential
            
            k1 = h * f(s1=s1, s2=s2, x=x, sigma=sigma)
            k2 = h * f(s1=s1, s2=s2, x=x + k1/2, sigma=sigma + h/2)
            k3 = h * f(s1=s1, s2=s2, x=x + k2/2, sigma=sigma + h/2)
            k4 = h * f(s1=s1, s2=s2, x=x + k3, sigma=sigma_next)
            k = (k1+2*k2+2*k3+k4)/6
            x_next = x + k
            
            xs_next.append(x_next)
        return xs_next

    def forward(
        self, noises: Tensor, fns: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        xs = [sigmas[0] * noise for noise in noises]
        # Denoise to sample
        for i in range(num_steps - 1):
            xs = self.step(xs, fns=fns, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa

        y1, y2 = xs
        y1, y2 = self.mixture - y2, self.mixture - y1

        y1_normalized, y2_normalized = least_squares_normalization(y1, y2, self.mixture)
        #y1_cons, y2_cons = enforce_mixture_consistency(self.mixture, torch.stack([y1_normalized, y2_normalized], dim=1))

        return [y1_normalized, y2_normalized]
        #return [y1, y2]
    

class HeunExtractor(Sampler):

    def __init__(self, mixture: torch.Tensor):
        super().__init__()
        self.mixture = mixture

    def compute_differential(self, s1, s2, x, sigma):
        grad_log_p1 = s1(x, sigma) - 1. * s2(self.mixture - x, sigma).mean(dim=0).unsqueeze(0)
        return -grad_log_p1

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        # Sigma steps
        S1 = lambda x, sigma: (fns[0](x, sigma=sigma) - x)/sigma
        S2 = lambda x, sigma: (fns[1](x, sigma=sigma) - x)/sigma

        xs_next = []
        for x, (s1,s2) in zip(xs, [(S1, S2), (S2, S1)]):
            # Derivative at sigma (∂x/∂sigma)
            d1 = self.compute_differential(s1=s1, s2=s2, x=x, sigma=sigma)

            # Euler step
            x_mid = x + d1 * (sigma_next - sigma)
            
            # Derivative at sigma next(∂x/∂sigma)
            d2 = self.compute_differential(s1=s1, s2=s2, x=x_mid, sigma=sigma_next)
            
            # Heun step
            x_next = x + 1/2 * (sigma_next - sigma) * (d1 + d2)

            xs_next.append(x_next)
        return xs_next

    def forward(
        self, noises: Tensor, fns: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        xs = [sigmas[0] * noise for noise in noises]
        # Denoise to sample
        for i in range(num_steps - 1):
            xs = self.step(xs, fns=fns, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa

        y1, y2 = xs
        y1, y2 = self.mixture - y2, self.mixture - y1

        y1_normalized, y2_normalized = least_squares_normalization(y1, y2, self.mixture)
        #y1_cons, y2_cons = enforce_mixture_consistency(self.mixture, torch.stack([y1_normalized, y2_normalized], dim=1))

        return [y1_normalized, y2_normalized]
        #eturn [y1, y2]
    

class CrawsonEulerExtractor(Sampler):

    def __init__(self, mixture: torch.Tensor):
        super().__init__()
        self.mixture = mixture

    def compute_differential(self, s1, s2, x, sigma):
        grad_log_p1 = s1(x, sigma) - 1. * s2(self.mixture - x, sigma).mean(dim=0).unsqueeze(0)
        return -grad_log_p1

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float, num_steps: int, s_churn=40., s_tmin=0., s_tmax=float('inf'), s_noise=1., is_last=False) -> List[Tensor]:
        # Sigma steps
        S1 = lambda x, sigma: (fns[0](x, sigma=sigma) - x)/sigma
        S2 = lambda x, sigma: (fns[1](x, sigma=sigma) - x)/sigma

        xs_next = []
        for x, (s1,s2) in zip(xs, [(S1, S2), (S2, S1)]):
            gamma = min(s_churn / (num_steps - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.
            #print("gamma = ", gamma)
            eps = torch.randn_like(x) * s_noise
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
            
            d = self.compute_differential(s1=s1, s2=s2, x=x, sigma=sigma_hat)
            dt = sigma_next - sigma_hat
            
            # Euler method
            x_next = x + d * dt

            xs_next.append(x_next)
        return xs_next

    def forward(
        self, noises: Tensor, fns: Callable, sigmas: Tensor, num_steps: int, s_churn: float = 20.0,
    ) -> Tensor:
        xs = [sigmas[0] * noise for noise in noises]
        
        # Denoise to sample
        for i in range(num_steps - 2):
            xs = self.step(xs, fns=fns, sigma=sigmas[i], sigma_next=sigmas[i + 1], num_steps=num_steps, s_churn=s_churn, is_last=False)  # type: ignore # noqa
        xs = self.step(xs, fns=fns, sigma=sigmas[-2], sigma_next=sigmas[-1], num_steps=num_steps, s_churn=s_churn, is_last=True)  # type: ignore # noqa

        y1, y2 = xs
        y1, y2 = self.mixture - y2, self.mixture - y1

        #y1_normalized, y2_normalized = least_squares_normalization(y1, y2, self.mixture)
        #y1_cons, y2_cons = enforce_mixture_consistency(self.mixture, torch.stack([y1_normalized, y2_normalized], dim=1))

        #return [y1_normalized, y2_normalized]
        return [y1, y2]


class CrawsonEulerSeparatorBASIS(Sampler):

    def __init__(self, mixture: torch.Tensor):
        super().__init__()
        self.mixture = mixture

    def compute_differential(self, s1, s2, x, sigma):
        grad_log_p1 = s1(x, sigma)
        return -grad_log_p1

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float, num_steps: int, s_churn=40., s_tmin=0., s_tmax=float('inf'), s_noise=1.) -> List[Tensor]:
        
        # Sigma steps
        def g(xs: List[Tensor]) -> Tensor:
            return torch.stack(xs, dim=0).sum(dim=0)
        
        
        xs_next = []
        for x, denoise_fn in zip(xs, fns):
            
            gamma = min(s_churn / (num_steps - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = sigma * (gamma + 1)
            
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
            
            d = (x - denoise_fn(x, sigma=sigma)) / sigma
            dt = sigma_next - sigma_hat
            
            # Euler method
            x_next = x + d * dt +  10 * (self.mixture - g(xs))
            xs_next.append(x_next)
    
        return xs_next

    def forward(
        self, noises: Tensor, fns: Callable, sigmas: Tensor, num_steps: int, s_churn: float = 20.0,
    ) -> Tensor:
        xs = [sigmas[0] * noise for noise in noises]
        
        # Denoise to sample
        for i in range(num_steps - 1):
            xs = self.step(xs, fns=fns, sigma=sigmas[i], sigma_next=sigmas[i + 1], num_steps=num_steps, s_churn=s_churn)  # type: ignore # noqa

        y1, y2 = xs
        y1, y2 = self.mixture - y2, self.mixture - y1

        #y1_normalized, y2_normalized = least_squares_normalization(y1, y2, self.mixture)
        #y1_cons, y2_cons = enforce_mixture_consistency(self.mixture, torch.stack([y1_normalized, y2_normalized], dim=1))
        
        return y1,y2
        return [y1_normalized, y2_normalized]

    
class HeunSampler(Sampler):

    def __init__(self, mixture: torch.Tensor):
        super().__init__()

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        # Input:
        # - xs: starting points
        # - fns: list of score fcts, one for every source
        # - sigma and sigma next are the noise levels
        
        # Gradient from score function
        grad1 = lambda x, sigma: (fns[0](x, sigma=sigma) - x)/sigma
        gard2 = lambda x, sigma: (fns[1](x, sigma=sigma) - x)/sigma
        
        gradients = [grad1, grad2]

        xs_next = []
        for x, gradient in zip(xs, gradients):
            # Derivative at sigma (∂x/∂sigma)
            d1 = gradient(x=x, sigma=sigma)

            # Euler step
            x_mid = x + d1 * (sigma_next - sigma)
            
            # Derivative at sigma next(∂x/∂sigma)
            d2 = gradient(x=x_mid, sigma=sigma_next)
            
            # Heun step
            x_next = x + 1/2 * (sigma_next - sigma) * (d1 + d2)

            xs_next.append(x_next)
        return xs_next

    def forward(
        self, noises: Tensor, fns: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        xs = [sigmas[0] * noise for noise in noises]
        
        # Denoise to sample
        for i in range(num_steps - 1):
            xs = self.step(xs, fns=fns, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa

        y1, y2 = xs
        
        return [y1, y2]

    
class CrawsonHeunExtractor(CrawsonEulerExtractor):
    def __init__(self, mixture: torch.Tensor):
        super().__init__(mixture)

    def step(
        self, 
        xs: List[Tensor], 
        fns: List[Callable], 
        sigma: float, 
        sigma_next: float, 
        num_steps: int, 
        s_churn=40., 
        s_tmin=0., 
        s_tmax=float('inf'), 
        s_noise=1.,
        is_last=False
    ) -> List[Tensor]:
        
        # Sigma steps
        S1 = lambda x, sigma: (fns[0](x, sigma=sigma) - x)/sigma
        S2 = lambda x, sigma: (fns[1](x, sigma=sigma) - x)/sigma

        xs_next = []
        for x, (s1,s2) in zip(xs, [(S1, S2), (S2, S1)]):
            
            gamma = min(s_churn / (num_steps - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = sigma * (gamma + 1)
            
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
            
            d = self.compute_differential(s1=s1, s2=s2, x=x, sigma=sigma_hat)
            dt = sigma_next - sigma_hat
            
            if is_last:
                # Euler method
                x_next = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                
                d_2 = self.compute_differential(s1=s1, s2=s2, x=x_2, sigma=sigma_next)
                d_prime = (d + d_2) / 2
                x_next = x + d_prime * dt
            
            xs_next.append(x_next)
        return xs_next


@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x
