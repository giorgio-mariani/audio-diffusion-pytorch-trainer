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


def enforce_mixture_consistency(
        mixture_waveforms,
        separated_waveforms,
):
    """Projection implementing mixture consistency in time domain.
      This projection makes the sum across sources of separated_waveforms equal
      mixture_waveforms and minimizes the unweighted mean-squared error between the
      sum across sources of separated_waveforms and mixture_waveforms. See
      https://arxiv.org/abs/1811.08521 for the derivation.
      Args:
        mixture_waveforms: Tensor of mixture waveforms in waveform format.
        separated_waveforms: Tensor of separated waveforms in source image format.
            mix_weights: None or Tensor of weights used for mixture consistency, shape
            should broadcast with denoised_waveforms. Overrides mix_weights_type.

      Returns:
        Projected separated_waveforms as a Tensor in source image format.
      """

    # Modify the source estimates such that they sum up to the mixture, where
    # the mixture is defined as the sum across sources of the true source
    # targets. Uses the least-squares solution under the constraint that the
    # resulting source estimates add up to the mixture.
    num_sources = separated_waveforms.shape[1]

    # Add a sources axis to mixture_spectrograms.
    mix = mixture_waveforms.unsqueeze(1)

    # mix is now of shape: (batch_size, 1, num_channels, samples)
    mix_estimate = torch.sum(separated_waveforms, dim=1, keepdims=True)

    # mix_estimate is of shape:
    # (batch_size, 1, num_channels, samples).
    mix_weights = torch.mean(torch.square(separated_waveforms), dim=[2, 3], keepdims=True)
    mix_weights = mix_weights / torch.sum(mix_weights, dim=1, keepdims=True)

    correction = mix_weights * (mix - mix_estimate)
    separated_waveforms = separated_waveforms + correction
    return tuple(separated_waveforms[:, i] for i in range(separated_waveforms.shape[1]))


def compute_guidance_grad(g_x: torch.Tensor, m: torch.Tensor, sigma: float, delta: float):
     m_tilde = m + torch.randn_like(m) * (2 * sigma)
     return (m - g_x) / delta**2


def g(xs: List[Tensor]) -> Tensor:
    return torch.stack(xs, dim=0).sum(dim=0)


class AEulerSeparator(Sampler):

    def __init__(self, mixture, delta: float = 1.0):
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
        for x, fn in zip(xs, fns):
            # Derivative at sigma (∂x/∂sigma)
            d = (x - fn(x, sigma=sigma)) / sigma - 8 * (self.mixture - g(xs)) / sigma

            # Euler method
            x_next = x + d * (sigma_next - sigma) #+ 0.2 * (self.mixture - g(xs))

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
    
class EulerSeparator(Sampler):

    def __init__(self, mixture, delta: float = 1.0):
        super().__init__()
        self.mixture = mixture
        self.delta = delta

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        xs_next = []
        for x, fn in zip(xs, fns):
            # Gradient at sigma (∂x/∂sigma)
            d = (x - fn(x, sigma=sigma)) / sigma

            # Euler method
            x_next = x + d * (sigma_next - sigma) #+ 0.2 * (self.mixture - g(xs))

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



class ADPM2Separator(Sampler):
    def __init__(self, mixture, rho: float = 1.0):
        super().__init__()
        self.rho = rho
        self.mixture = mixture

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)

        xs_mid, xs_next = [], []
        for x, fn in zip(xs, fns):
            # Derivative at sigma (∂x/∂sigma)
            d = (x - fn(x, sigma=sigma)) / sigma
            # Denoise to midpoint
            x_mid = x + d * (sigma_mid - sigma)# - 8 * (self.mixture - g(xs)) / sigma
            xs_mid.append(x_mid)

        for x, x_mid, fn in zip(xs, xs_mid, fns):
            # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
            d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid #- 8 * (self.mixture - g(xs_mid)) / sigma_mid

            # Denoise to next
            x = x + d_mid * (sigma_down - sigma)

            # Add randomness
            x_next = x + torch.randn_like(x) * sigma_up + 0.2*(self.mixture - g(xs))
            xs_next.append(x_next)

        #print(tuple(x.norm() for x in xs_next))
        return xs_next

    def forward(
        self, noises: List[Tensor], fns: List[Callable], sigmas: Tensor, num_steps: int
    ) -> List[Tensor]:
        xs = [sigmas[0] * noise for noise in noises]
        # Denoise to sample
        for i in range(num_steps - 1):
            xs = self.step(xs, fns, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return xs


class AEulerSeparatorSymmetric(Sampler):
    def __init__(self, mixture, delta: float = 1.0):
        super().__init__()
        self.mixture = mixture
        self.delta = delta

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        # Sigma steps
        S1 = lambda x, sigma: (fns[0](x, sigma=sigma) - x)/sigma**2
        S2 = lambda x, sigma: (fns[1](x, sigma=sigma) - x)/sigma**2

        # Derivative at sigma (∂x/∂sigma)
        x1, x2 = xs
        m = self.mixture
        epsilon = 0.0#torch.randn(size=[1, 1, 1], device=m.device)
        grad_log_p1 = S1(x1, sigma) - 1.2 * S2(m - x1 + 2*sigma*epsilon, sigma).mean(dim=0).unsqueeze(0)
        grad_log_p2 = S2(x2, sigma) - 1.2 * S1(m - x2 + 2*sigma*epsilon, sigma).mean(dim=0).unsqueeze(0)

        # Euler method
        x1_next = x1 + (sigma_next - sigma) * -grad_log_p1 * sigma
        x2_next = x2 + (sigma_next - sigma) * -grad_log_p2 * sigma

        # Add randomness
        #x_next = x_next + torch.randn_like(x) * sigma_up
        return [x1_next, x2_next]

    def forward(
        self, noises: List[Tensor], fns: List[Callable], sigmas: Tensor, num_steps: int
    ) -> List[Tensor]:
        xs = [sigmas[0] * noise for noise in noises]
        # Denoise to sample
        for i in range(num_steps - 1):
            xs = self.step(xs, fns, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return xs


class ADPM2Extractor(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    def __init__(self, mixture: torch.Tensor, rho: float = 1.0):
        super().__init__()
        self.rho = rho
        self.mixture = mixture

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def compute_differential(self, s1, s2, x, sigma):
        grad_log_p1 = s1(x, sigma) - 1. * s2(self.mixture - x, sigma).mean(dim=0).unsqueeze(0)
        return -grad_log_p1 * sigma

    def step(self, xs: List[Tensor], fns: List[Callable], sigma: float, sigma_next: float) -> List[Tensor]:
        # Sigma steps
        S1 = lambda x, sigma: (fns[0](x, sigma=sigma) - x)/sigma**2
        S2 = lambda x, sigma: (fns[1](x, sigma=sigma) - x)/sigma**2

        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)

        xs_next = []
        for x, (s1,s2) in zip(xs, [(S1, S2), (S2, S1)]):
            # Derivative at sigma (∂x/∂sigma)
            d = self.compute_differential(s1=s1, s2=s2, x=x, sigma=sigma)

            # Denoise to midpoint
            x_mid = x + d * (sigma_mid - sigma)

            # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
            d_mid = self.compute_differential(s1=s1, s2=s2, x=x_mid, sigma=sigma_mid)

            # Denoise to next
            x = x + d_mid * (sigma_down - sigma)

            # Add randomness
            x_next = x + torch.randn_like(x) * sigma_up
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

        #y1_normalized, y2_normalized = least_squares_normalization(y1, y2, self.mixture)
        #y1_cons, y2_cons = enforce_mixture_consistency(self.mixture, torch.stack([y1_normalized, y2_normalized], dim=1))

        #return [y1_normalized, y2_normalized]
        return [y1, y2]
    

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

        #y1_normalized, y2_normalized = least_squares_normalization(y1, y2, self.mixture)
        #y1_cons, y2_cons = enforce_mixture_consistency(self.mixture, torch.stack([y1_normalized, y2_normalized], dim=1))

        #return [y1_normalized, y2_normalized]
        return [y1, y2]
    

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

        #y1_normalized, y2_normalized = least_squares_normalization(y1, y2, self.mixture)
        #y1_cons, y2_cons = enforce_mixture_consistency(self.mixture, torch.stack([y1_normalized, y2_normalized], dim=1))

        #return [y1_normalized, y2_normalized]
        return [y1, y2]
    

def least_squares_normalization(y1,y2,m):
    y1np, y2np = y1.view(-1).cpu().numpy(), y2.view(-1).cpu().numpy()
    y = np.stack([y1np, y2np], axis=1)
    a = np.linalg.lstsq(y, m.view(-1,1).cpu().numpy())
    a,_,_,_ = a
    a1, a2 = a.reshape(-1).tolist()
    return a1*y1, a2*y2


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
    denoise_fn,
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
    sigmas = torch.cat([denoise_fn.round_sigma(sigmas), torch.zeros_like(sigmas[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * sigmas[0]
    for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(s_churn / num_steps, sqrt(2) - 1) if s_min <= sigma <= s_max else 0
        t_hat = denoise_fn.round_sigma(sigma + gamma * sigma) #What is it?
        x_hat = x_cur + (t_hat ** 2 - sigma ** 2).sqrt() * s_noise * randn_like(x_cur)

        # Euler step.
        denoised = denoise_fn(x_hat, sigma=t_hat).to(torch.float64)
        d = (x_hat - denoised) / t_hat
        x_next = x_hat + (sigma_next - t_hat) * d

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = denoise_fn(x_next, sigma=sigma_next).to(torch.float64)
            d_prime = (x_next - denoised) / sigma_next
            x_next = x_hat + (sigma_next - t_hat) * (0.5 * d + 0.5 * d_prime)


@torch.no_grad()
def separate(
    denoise_fns: List[Callable],
    separator,
    noises,
    num_steps: int = 100,
    sigma_schedule: Schedule = None,
    device: torch.device = torch.device("cpu"),
):
    sigma_schedule = KarrasSchedule(1e-2, 1.0, rho=9.0) if sigma_schedule is None else sigma_schedule
    sigma = sigma_schedule(num_steps, device)
    return separator(noises=noises, fns=denoise_fns, sigmas=sigma, num_steps=num_steps)


def separate_dataset(
    dataset: SeparationDataset,
    denoise_fns: List[Callable],
    save_path: str = "evaluation_results",
    device: Optional[torch.device] = None,
    separation_steps: int = 100,
    separator = ADPM2Separator,
    sigma_schedule: Schedule = None,
):
    # get arguments
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    device = torch.device(device)
    sample_rate = dataset.sample_rate

    # convert paths
    save_path = Path(save_path)
    if save_path.exists() and not len(list(save_path.glob("*"))) == 0:
        raise ValueError(f"Path {save_path} already exists!")

    # get samples
    loader = DataLoader(dataset, batch_size=None, num_workers=8)

    # main loop
    save_path.mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm.tqdm(loader)):

        # load audio tracks
        tracks = batch
        print(f"chunk {batch_idx+1} out of {len(dataset)}")
        # generate mixture
        tracks = [track.unsqueeze(0).to(device) for track in tracks]
        mixture = sum(tracks)

        _, _, num_samples = mixture.shape

        seps = separate(
            denoise_fns=denoise_fns,
            separator=separator(mixture=mixture),
            sigma_schedule=sigma_schedule,
            num_steps=separation_steps,
            noises = [torch.randn(1, 1, num_samples).to(device) for _ in range(len(denoise_fns))]
        )

        chunk_path = save_path / f"{batch_idx}"
        chunk_path.mkdir(parents=True)


        # save separated audio
        save_separation(
            separated_tracks=[sep.squeeze(0) for sep in seps],
            original_tracks=[track.squeeze(0) for track in tracks],
            sample_rate=sample_rate,
            chunk_path=chunk_path,
        )


def save_separation(
    separated_tracks: List[torch.Tensor],
    original_tracks: List[torch.Tensor],
    sample_rate: int,
    chunk_path: Path,
):
    assert_is_audio(*original_tracks, *separated_tracks)
    #assert original_1.shape == original_2.shape == separation_1.shape == separation_2.shape
    assert len(original_tracks) == len(separated_tracks)
    for i, (ori, sep) in enumerate(zip(original_tracks, separated_tracks)):
        torchaudio.save(chunk_path / f"ori{i+1}.wav", ori.cpu(), sample_rate=sample_rate)
        torchaudio.save(chunk_path / f"sep{i+1}.wav", sep.cpu(), sample_rate=sample_rate)