from typing import List, Optional, Callable, Tuple

import torch
from torch import Tensor
from math import sqrt

from audio_diffusion_pytorch.diffusion import AEulerSampler, ADPM2Sampler, Diffusion, KarrasSchedule, Sampler, Schedule
from audio_diffusion_pytorch.model import AudioDiffusionModel
from audio_diffusion_pytorch.utils import default, exists


def compute_guidance_grad(g_x: torch.Tensor, m: torch.Tensor, sigma: float, delta: float):
     m_tilde = m + torch.randn_like(m) * (2 * sigma)
     return (m_tilde - g_x) / delta**2


def g(xs: List[Tensor]) -> Tensor:
    return torch.stack(xs, dim=0).sum(dim=0)


class AEulerSeparator(Sampler):

    def __init__(self, mixture, delta):
        super().__init__()
        self.mixture = mixture
        self.delta = delta

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float]:
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        return sigma_up, sigma_down

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        # Sigma steps
        sigma_up, sigma_down = self.get_sigmas(sigma, sigma_next)
        xs_next = []
        guid_grad = compute_guidance_grad(g_x=g(xs), m=self.mixture, sigma=sigma, delta=self.delta)

        for x, fn in zip(xs,fns):
            # Derivative at sigma (∂x/∂sigma)
            d = (x - fn(x, sigma=sigma)) / sigma #+ 0.2 * self.delta**2/(sigma_next - sigma) * guid_grad

            # Euler method
            x_next = x + d * (sigma_next - sigma) + 0.2 * (self.mixture - g(xs))

            # Add randomness
            #x_next = x_next + torch.randn_like(x) * sigma_up
            xs_next.append(x_next)
        return xs_next

    def forward(
        self, noises: List[Tensor], fns: List[Callable], sigmas: Tensor, num_steps: int
    ) -> List[Tensor]:
        xs = [sigmas[0] * noise for noise in noises]
        # Denoise to sample
        for i in range(num_steps - 1):
            xs = self.step(xs, fns, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return xs


class KarrasSeparator(Sampler):
    """https://arxiv.org/abs/2206.00364 algorithm 2"""

    def __init__(
        self,
        mixture: torch.Tensor,
        delta: float,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_churn: float = 0.0,
        s_noise: float = 1.0,
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn
        self.delta = delta
        self.mixture = mixture

    def step(
        self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float, gamma: float, step: int,
    ) -> Tensor:
        """Algorithm 2 (step)"""
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma
        m = self.mixture

        xs_hat = []
        for x in xs:
            # Add noise to move from sigma to sigma_hat
            epsilon = self.s_noise * torch.randn_like(x)
            x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon
            xs_hat.append(x_hat)

        xs_next = []
        guid_grad_hat = compute_guidance_grad(g_x=g(xs_hat), m=m, sigma=sigma_hat, delta=self.delta)
        for x_hat, fn in zip(xs_hat, fns):
            # Evaluate ∂x/∂sigma at sigma_hat
            d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat# + sigma_hat * guid_grad_hat

            # Take euler step from sigma_hat to sigma_next
            x_next = x_hat + (sigma_next - sigma_hat) * d
            xs_next.append(x_next)

        # Second order correction
        if sigma_next != 0:
            guid_grad_next = compute_guidance_grad(g_x=g(xs_next), m=m, sigma=sigma_next, delta=self.delta)
            for i, (x_next, fn) in enumerate(zip(xs_next,fns)):
                model_out_next = fn(x_next, sigma=sigma_next)
                d_prime = (x_next - model_out_next) / sigma_next + sigma_next * guid_grad_next
                xs_next[i] = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)

        if step % 20 == 0:
            print("distance to mixture:", torch.norm(g(xs_next) - m).item())
            print("likelihood norm:", torch.norm(guid_grad_hat).item())
            print("")
            # plot_waves(xs[0].cpu(), xs[1].cpu())

        return xs_next

    def forward(
        self, noises: List[Tensor], fns: List[Callable], sigmas: Tensor, num_steps: int
    ) -> Tensor:
        assert len(noises) == len(fns)

        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / num_steps, sqrt(2) - 1),
            0.0,
        )

        # Denoise to sample
        xs = [sigmas[0] * noise for noise in noises]
        for i in range(num_steps - 1):
            xs = self.step(
                xs, fns=fns, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i], step=i
            )

        return xs


def edm_sampler(
    fn,
    latents,
    randn_like=torch.randn_like,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7.0,
    s_churn: float = 0.0,
    s_min: float = 0.0,
    s_max: float = float('inf'),
    s_noise: float = 1.0,
):

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    sigmas = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = torch.cat([fn.round_sigma(sigmas), torch.zeros_like(sigmas[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * sigmas[0]
    for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(s_churn / num_steps, sqrt(2) - 1) if s_min <= sigma <= s_max else 0
        t_hat = fn.round_sigma(sigma + gamma * sigma) #What is it?
        x_hat = x_cur + (t_hat ** 2 - sigma ** 2).sqrt() * s_noise * randn_like(x_cur)

        # Euler step.
        denoised = fn(x_hat, sigma=t_hat).to(torch.float64)
        d = (x_hat - denoised) / t_hat
        x_next = x_hat + (sigma_next - t_hat) * d

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = fn(x_next, sigma=sigma_next).to(torch.float64)
            d_prime = (x_next - denoised) / sigma_next
            x_next = x_hat + (sigma_next - t_hat) * (0.5 * d + 0.5 * d_prime)

#
# @torch.no_grad()
# def separate(
#         model1,
#         model2,
#         mixture,
#         device: torch.device = torch.device("cuda"),
#         num_steps: int = 202,
# ):
#     batch, in_channels = 1, 1
#     samples = mixture.shape[-1]
#
#     m = torch.tensor(mixture).to(device)
#     models = [model1.model, model2.model]
#
#     for model in models:
#         model.to(device)
#
#     # schedule = lambda num_steps,device: torch.flip(torch.sort(diffusion_sigma_distribution(num_steps*10, device))[0][::10],dims=[0])
#     # schedule = lambda num_steps, device: torch.arange(0.6, 0.0, -0.6/num_steps,device=device)
#     schedule = KarrasSchedule(sigma_min=1e-4, sigma_max=1.0, rho=8.0)
#
#     diffusion_separator = DiffusionSeparator(
#         [model.diffusion for model in models],
#         samplers=[ADPM2Sampler(rho=1.0), ADPM2Sampler(rho=1.0)],
#         sigma_schedules=[schedule, schedule],
#         num_steps=num_steps
#     )
#
#     noises = [torch.randn_like(m).to(device), torch.randn_like(m).to(device)]
#     return diffusion_separator.forward(m, noises)