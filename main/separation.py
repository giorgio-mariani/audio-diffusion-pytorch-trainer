import abc
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Mapping
from tqdm import tqdm
from scipy import integrate

import numpy as np
import torch
import torchaudio
from torch import Tensor
from math import sqrt

from audio_diffusion_pytorch.diffusion import Sampler, KarrasSchedule, Schedule
from torch.utils.data import DataLoader
from torch import nn
from torchdiffeq import odeint
import main.utils as utils

from main.dataset import assert_is_audio, SeparationDataset
from main.module_base import Model

class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    def separate(mixture, num_steps) -> Mapping[str, torch.Tensor]:
        ...
    
    
class ContextualSeparator(Separator):
    def __init__(self, model: Model, stems: List[str], sigma_schedule: Schedule, **kwargs):
        super().__init__()
        self.model = model
        self.stems = stems
        self.sigma_schedule = sigma_schedule
        self.separation_kwargs = kwargs
    
    def separate(self, mixture: torch.Tensor, num_steps:int = 100):
        device = self.model.device
        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape
        
        y = separate_mixture(
            mixture=mixture,
            denoise_fn=self.model.model.diffusion.denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(self.stems), length_samples).to(device),
            **self.separation_kwargs,
        )
        return {stem:y[:,i,:] for i,stem in enumerate(self.stems)}

    def separate_with_hint(
        self,
        mixture: torch.Tensor,
        source_with_hint: torch.Tensor,
        mask: torch.Tensor,
        num_steps:int = 100,
        ):
        print(self.separation_kwargs)
        
        device = self.model.device
        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape

        y = inpaint_mixture(
            source = source_with_hint,
            mask = mask,
            mixture = mixture,
            fn = self.model.model.diffusion.denoise_fn,
            sigmas = self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(self.stems), length_samples).to(device),
            **self.separation_kwargs,
        )

        return {stem:y[:,i:i+1,:] for i,stem in enumerate(self.stems)}


class IndependentSeparator(Separator):
    def __init__(self, stem_to_model: Mapping[str, Model], sigma_schedule, **kwargs):
        super().__init__()
        self.stem_to_model = stem_to_model
        self.separation_kwargs = kwargs
        self.sigma_schedule = sigma_schedule

    
    def separate(self, mixture: torch.Tensor, num_steps: int):
        stems = self.stem_to_model.keys()
        models = [self.stem_to_model[s] for s in stems]
        fns = [m.model.diffusion.denoise_fn for m in models]
        
        # get device of models
        devices = {m.device for m in models}
        assert len(devices) == 1, devices
        (device,) = devices
        
        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape

        def denoise_fn(x, sigma):
            xs = [x[:, i:i+1] for i in range(4)]
            xs = [fn(x,sigma=sigma) for fn,x in zip(fns, xs)]
            return torch.cat(xs, dim=1)
        
        y = separate_mixture(
            mixture=mixture,
            denoise_fn=denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(stems), length_samples).to(device),
            **self.separation_kwargs,
        )
        return {stem:y[:,i:i+1,:] for i, stem in enumerate(stems)}

# Algorithms ------------------------------------------------------------------


def differential_with_dirac(x, sigma, denoise_fn, mixture, source_id=0):
    num_sources = x.shape[1]
    print(source_id)
    # + torch.randn_like(self.mixture) * sigma
    x[:, [source_id], :] = mixture - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
    score = (x - denoise_fn(x, sigma=sigma)) / sigma
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s - score[:, source_id] for s in scores]
    return torch.stack(ds, dim=1)


def differential_with_gaussian(x, sigma, denoise_fn, mixture, gamma_fn=None):
    gamma = sigma if gamma_fn is None else gamma_fn(sigma)
    d = (x - denoise_fn(x, sigma=sigma)) / sigma 
    d = d - sigma / (2 * gamma ** 2) * (mixture - x.sum(dim=[1], keepdim=True)) 
    #d = d - 8/sigma * (mixture - x.sum(dim=[1], keepdim=True)) 
    return d


@torch.no_grad()
def separate_mixture(
    mixture: torch.Tensor, 
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    noises: Optional[torch.Tensor],
    differential_fn: Callable = differential_with_dirac,
    s_churn: float = 20.0, # > 0 to add randomness
    use_heun: bool = False,
    num_resamples: int = 1
):      
    # Set initial noise
    x = sigmas[0] * noises # [batch_size, num-sources, sample-length]
    
    for i in range(len(sigmas) - 1):
        for r in range(num_resamples):
            x = step(
                x,
                i,
                mixture, 
                denoise_fn,
                sigmas,
                differential_fn,
                s_churn,
                use_heun,
            )
            if r < num_resamples - 1:
                sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                x = x + sigma * torch.randn_like(x)
    
    return x.cpu().detach()



@torch.no_grad()
def step(
    x: torch.Tensor,
    i: int,
    mixture: torch.Tensor, 
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    differential_fn: Callable = differential_with_dirac,
    s_churn: float = 0.0, # > 0 to add randomness
    use_heun: bool = False,
    source_id: int = 0,
    gradient_mean: bool = False,
    **kwargs
):      
    sigma, sigma_next = sigmas[i], sigmas[i+1]
    print(f"{gradient_mean=}")

    # Inject randomness
    gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
    sigma_hat = sigma * (gamma + 1)            
    x_hat = x + torch.randn_like(x) * (sigma_hat ** 2 - sigma ** 2) ** 0.5

    # Compute conditioned derivative
    if gradient_mean:
        print("bau")
        d_list = []
        for i in range(4):
            d_list.append(differential_fn(mixture=mixture, x=x_hat, sigma=sigma_hat, denoise_fn=denoise_fn, source_id=i))
        d_tensor = torch.stack(d_list)
        d = d_tensor.mean(0)
    else: 
        d = differential_fn(mixture=mixture, x=x_hat, sigma=sigma_hat, denoise_fn=denoise_fn, source_id=source_id)

    # Update integral
    if not use_heun or sigma_next == 0.0:
        # Euler method
        x = x_hat + d * (sigma_next - sigma_hat) 
    else:
        # Heun's method
        x_2 = x_hat + d * (sigma_next - sigma_hat)
        d_2 = differential_fn(mixture=mixture, x=x_2, sigma=sigma_next, denoise_fn=denoise_fn)
        d_prime = (d + d_2) / 2
        x = x + d_prime * (sigma_next - sigma_hat)
    if not gradient_mean:
        x[:, [source_id], :] = mixture - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
    return x


@torch.no_grad()
def inpaint_mixture(
    source: Tensor,
    mask: Tensor,
    mixture: Tensor,
    fn: Callable,
    sigmas: Tensor,
    noises: Tensor,
    differential_fn: Callable = differential_with_dirac,
    num_resamples: int = 1,
    s_churn: float = 20.0, # > 0 to add randomness
    use_heun: bool = False,
    source_id: int = 0,
    gradient_mean: bool = False,
    **kwargs
) -> Tensor:
    
    print(f"{gradient_mean=}")
    
    x = sigmas[0] * noises
    num_sources = source.shape[1]
    if source_id == -1:
        variable_source_id = True
    else:
        variable_source_id = False

    for i in tqdm(range(len(sigmas) - 1)):
        if variable_source_id:
            source_id = torch.randint(high=num_sources, size=(1,)).tolist()[0]
        # Noise source to current noise level
        num_sources = x.shape[1]
        source_noisy = source + sigmas[i] * torch.randn_like(source)
        for r in range(num_resamples):
            # Merge noisy source and current then denoise
            x = source_noisy * mask + x * mask.logical_not()
            x = step(x, i, mixture=mixture, denoise_fn=fn, 
                    sigmas=sigmas, s_churn=s_churn, use_heun=use_heun, 
                     source_id=source_id, gradient_mean=gradient_mean, **kwargs)  # type: ignore # noqa
            # Renoise if not last resample step
            if r < num_resamples - 1:
                sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                x = x + sigma * torch.randn_like(x)

    return source * mask + x * mask.logical_not()



def separate_basis_original(
    denoise_fns: List[Callable],
    mixture: torch.Tensor,
    step_lr: float = 0.00003,
    num_steps_each: int = 100,
):
    num_sources = len(denoise_fns)
    xs = []
    for _ in range(num_sources):
        xs.append(torch.nn.Parameter(torch.Tensor(*mixture.shape).uniform_()).cuda())

    # Noise amounts
    sigmas = np.array(
        [1., 0.59948425, 0.35938137, 0.21544347, 0.12915497, 0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01]
    )

    for idx, sigma in enumerate(sigmas):
        lambda_recon = 1. / (sigma ** 2)
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for step in range(num_steps_each):
            noises = []
            for _ in range(num_sources):
                noises.append(torch.randn_like(xs[0]) * np.sqrt(step_size * 2))

            grads = []
            for i, fn in enumerate(denoise_fns):
                grad_logp = (fn(xs[i], sigma=sigma) - xs[i]) / sigma**2
                grads.append(grad_logp.detach())

            recon_loss = torch.norm(torch.flatten(sum(xs) - mixture)) ** 2
            recon_grads = torch.autograd.grad(recon_loss, xs)

            for i in range(num_sources):
                xs[i] = xs[i] + step_size * grads[i] + (-step_size * lambda_recon * recon_grads[i].detach()) + noises[i]

    return [torch.clamp(xs[i], -1.0, 1.0).detach().cpu() for i in range(num_sources)]
# -----------------------------------------------------------------------------


@torch.no_grad()
def separate_dataset(
    dataset: SeparationDataset,
    separator: Separator,
    num_steps: int,
    save_path: str = "evaluation_results",
):

    # convert paths
    save_path = Path(save_path)
    if save_path.exists() and not len(list(save_path.glob("*"))) == 0:
        raise ValueError(f"Path {save_path} already exists!")

    # get samples
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    # main loop
    save_path.mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(loader)):

        # load audio tracks
        tracks = batch
        print(f"chunk {batch_idx+1} out of {len(dataset)}")
        
        # generate mixture
        mixture = sum(tracks)
        seps = separator.separate(mixture=mixture, num_steps=num_steps)

        chunk_path = save_path / f"{batch_idx}"
        chunk_path.mkdir(parents=True)

        # save separated audio
        save_separation(
            separated_tracks=[sep for sep in seps.values()],
            original_tracks=[track.squeeze(0) for track in tracks],
            sample_rate=dataset.sample_rate,
            chunk_path=chunk_path,
        )


def save_separation(
    separated_tracks: List[torch.Tensor],
    original_tracks: List[torch.Tensor],
    sample_rate: int,
    chunk_path: Path,
):
    print([t[0].shape for t in original_tracks])
    print([t[0].shape for t in original_tracks])

    assert_is_audio(*original_tracks, *separated_tracks)
    #assert original_1.shape == original_2.shape == separation_1.shape == separation_2.shape
    assert len(original_tracks) == len(separated_tracks)
    for i, (ori, sep) in enumerate(zip(original_tracks, separated_tracks)):
        torchaudio.save(chunk_path / f"ori{i+1}.wav", ori.cpu(), sample_rate=sample_rate)
        torchaudio.save(chunk_path / f"sep{i+1}.wav", sep.cpu(), sample_rate=sample_rate)


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
        mixture_waveforms: Tensor of mixture waveforms [batch_size, 1, sample_length].
        separated_waveforms: Tensor of source waveforms [batch_size, num_sources, sample_length].

      Returns:
        Projected separated_waveforms as a Tensor with the same shape as separated_waveforms.
      """

    # Modify the source estimates such that they sum up to the mixture, where
    # the mixture is defined as the sum across sources of the true source
    # targets. Uses the least-squares solution under the constraint that the
    # resulting source estimates add up to the mixture.
    num_sources = separated_waveforms.shape[1]

    # Add a sources axis.
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


def least_squares_normalization(ys: List[torch.Tensor], mixture: torch.Tensor):
    # ys must be a list of tensors with shape [1,1, N]
    ys_np = [y.view(-1).cpu().numpy() for y in ys]
    y = np.stack(ys_np, axis=1)
    
    #compute optimal coefficients given the mixture
    a,_,_,_ = np.linalg.lstsq(y, mixture.view(-1, 1).cpu().numpy())
    alphas = a.reshape(-1).tolist()
    return [a*y for a,y in zip(alphas, ys)]



def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    #return (x - denoised) / utils.append_dims(sigma, x.ndim)
    return (x - denoised) / sigma

@torch.no_grad()
def log_likelihood_crawford(model, x, sigma_min, sigma_max, atol=1e-4, rtol=1e-4):
    #print(f"{x.shape=}")
    s_in = x.new_ones([x.shape[0]])
    #print(f"{s_in.shape=}")
    v = torch.randint_like(x, 2) * 2 - 1
    #print(f"{v=}")
    fevals = 0
    def ode_fn(sigma, x):
        print("----------------------------------------------------")
        #print(f"{sigma=}")
        nonlocal fevals
        with torch.enable_grad():
            fevals += 1
            y = x.detach().requires_grad_()
            #print(f"{x.shape=}")
            denoised = model(y, sigma * s_in)
            d = to_d(y, sigma, denoised)
            #print(f"{d=}")
            grad = torch.autograd.grad((d * v).sum(), y, retain_graph=True)[0]
            #print(f"{grad.shape=}")
            d_ll = (v * grad).flatten(1).sum(1)
            #print(f"{d_ll=}")
            print(d_ll)
        return d.detach(), d_ll
    x_min = x#, x.new_zeros([x.shape[0]])
    print(f"{x_min=}")
    t = x.new_tensor([sigma_min, sigma_max])
    print(f"{t=}")
    sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
    #print(f"{sol=}")
    latent, delta_ll = sol[0][-1], sol[1][-1]
    ll_prior = torch.distributions.Normal(0, sigma_max).log_prob(latent).flatten(1).sum(1)
    return ll_prior + delta_ll, {'fevals': fevals}

@torch.no_grad()
def log_likelihood_irene(model, x, sigma_min, sigma_max, atol=1e-4, rtol=1e-4):
    #print(f"{x.shape=}")
    eps = torch.randint_like(x.reshape((1, x.numel())), 2) * 2 - 1
    #print(f"{eps.shape=}")
    fevals = 0
    def ode_fn(sigma, x):
        nonlocal fevals
        with torch.enable_grad():
            x = x[0].detach().requires_grad_()
            #print(f"{x.shape=} bau")
            #print(f"{sigma=}")
            denoised = model(x, sigma.unsqueeze(0))
            d = to_d(x, sigma, denoised).reshape((x.numel(), 1))
            #print(f"{d.shape=}")
            fevals += 1
            grad = torch.autograd.grad(eps @ d, x)[0].reshape((x.numel(), 1))
            #print(f"{grad.shape=}")
            d_ll = (eps @ grad).mean()
        return d.detach(), d_ll
    x_min = x, x.new_zeros([x.shape[0]])
    t = x.new_tensor([sigma_min, sigma_max])
    #print(f"{t.shape=}")
    #print(f"{x_min[0].shape=}")
    sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
    latent, delta_ll = sol[0][-1], sol[1][-1]
    ll_prior = torch.distributions.Normal(0, sigma_max).log_prob(latent).flatten(1).sum(1)
    return ll_prior + delta_ll, {'fevals': fevals}, sol

def log_likelihood_song(model, data, sigma_max, hutchinson_type="Gaussian"):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.
    Args:
      model: A score model.
      data: A PyTorch tensor.
    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
        shape = data.shape
        if hutchinson_type == 'Gaussian':
            epsilon = torch.randn_like(data)
        elif hutchinson_type == 'Rademacher':
            epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        def ode_func(t, x):
            sample = utils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
            vec_t = torch.ones(sample.shape[0], device=sample.device) * t
            drift = utils.to_flattened_numpy(drift_fn_karras(model, sample, vec_t))
            logp_grad = utils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
            out = np.concatenate([drift, logp_grad], axis=0)
            return out

        init = np.concatenate([utils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
        
        solution = integrate.solve_ivp(ode_func, (1e-5, sigma_max), init, rtol=1e-5, atol=1e-5, method="RK45")
        nfe = solution.nfev
        zp = solution.y[:, -1]
        #print(f"{zp.shape=}")
        z = utils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
        #print(f"{z.shape=}")
        delta_logp = utils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
        #print(f"{delta_logp=}")
        prior_logp = torch.distributions.Normal(0, sigma_max).log_prob(z).sum()
        #print(f"{prior_logp=}")
        bpd = -(prior_logp + delta_logp) / np.log(2)
        N = np.prod(shape[1:])
        bpd = bpd / N
        return bpd, z, nfe


def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn_karras(model, xx, tt))(x, t, noise)

def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
    return div_fn

def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = utils.get_score_fn(sde, model, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

def drift_fn_karras(model, x, sigma):
    """The drift function of the reverse-time SDE."""
    denoised = model(x, sigma)
    return (x - denoised) / sigma