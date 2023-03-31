import abc
from typing import List, Optional, Callable, Mapping
from tqdm import tqdm

import torch
from torch import Tensor
from math import sqrt

from audio_diffusion_pytorch.diffusion import Schedule

from main.module_base import Model



# Generators

class Generator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def generate(self, batch_size: int, length_samples: int, num_steps: int) -> Mapping[str, torch.Tensor]:
        ...

class ContextualGenerator(Generator):
    ...
    # TODO


def differential_generation_ctc(x, sigma, denoise_fn, mixture=None):
    num_sources = x.shape[1]
    # x[:, [-1], :] = (x.sum(dim=1, keepdim=True) - x[:, [-1], :])
    score = (x - denoise_fn(x, sigma=sigma)) / sigma
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s for s in scores] # [s + score[:, -1] for s in scores]
    return torch.stack(ds, dim=1)


class ContextRegularizedGenerator(Generator):
    def __init__(self, stem_to_model: Mapping[str, Model], sigma_schedule, **kwargs):
        super().__init__()
        self.stem_to_model = stem_to_model
        self.separation_kwargs = kwargs
        self.sigma_schedule = sigma_schedule

    def generate(self, batch_size: int, length_samples: int, num_steps: int):
        stems = self.stem_to_model.keys()
        models = [self.stem_to_model[s] for s in stems]
        fns = [m.model.diffusion.denoise_fn for m in models]

        # get device of models
        devices = {m.device for m in models}
        assert len(devices) == 1, devices
        (device,) = devices

        def denoise_fn(x, sigma):
            xs = [x[:, i:i + 1] for i in range(len(stems))]
            xs = [fn(x, sigma=sigma) for fn, x in zip(fns, xs)]
            return torch.cat(xs, dim=1)

        y = sample(
            denoise_fn=denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(stems), length_samples).to(device),
            differential_fn=differential_generation_ctc,
            **self.separation_kwargs,
        )
        return {stem: y[:, i:i + 1, :] for i, stem in enumerate(stems)}

    def generate_partial(
            self,
            source: torch.Tensor,
            mask: torch.Tensor,
            num_steps: int = 100,
    ):
        stems = self.stem_to_model.keys()
        models = [self.stem_to_model[s] for s in stems]
        fns = [m.model.diffusion.denoise_fn for m in models]

        # get device of models
        devices = {m.device for m in models}
        assert len(devices) == 1, devices
        (device,) = devices

        source = source.to(device)
        mask = mask.to(device)
        batch_size, _, length_samples = source.shape

        def denoise_fn(x, sigma):
            xs = [x[:, i:i + 1] for i in range(len(stems))]
            xs = [fn(x, sigma=sigma) for fn, x in zip(fns, xs)]
            return torch.cat(xs, dim=1)

        y = inpaint(
            source=source,
            mask=mask,
            denoise_fn=denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(self.stems), length_samples).to(device),
            differential_fn=differential_generation_ctc,
            **self.separation_kwargs,
        )

        return {stem: y[:, i:i + 1, :] for i, stem in enumerate(self.stems)}

# Separators


def differential_separation_msdm_dirac(x, sigma, denoise_fn, mixture, source_id=0):
    num_sources = x.shape[1]
    # + torch.randn_like(self.mixture) * sigma
    x[:, [source_id], :] = mixture - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
    score = (x - denoise_fn(x, sigma=sigma)) / sigma
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s - score[:, source_id] for s in scores]
    return torch.stack(ds, dim=1)


def differential_separation_gaussian(x, sigma, denoise_fn, mixture, gamma_fn=None):
    gamma = sigma if gamma_fn is None else gamma_fn(sigma)
    d = (x - denoise_fn(x, sigma=sigma)) / sigma
    d = d - sigma / (2 * gamma ** 2) * (mixture - x.sum(dim=[1], keepdim=True))
    # d = d - 8/sigma * (mixture - x.sum(dim=[1], keepdim=True))
    return d


class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def separate(self, mixture, num_steps) -> Mapping[str, torch.Tensor]:
        ...


class ContextualSeparator(Separator):
    def __init__(self, model: Model, stems: List[str], sigma_schedule: Schedule,
                 differential : 'str' = 'dirac', **kwargs):
        super().__init__()
        self.model = model
        self.stems = stems
        self.sigma_schedule = sigma_schedule
        if differential == 'dirac':
            self.differential_fn = differential_separation_msdm_dirac
        elif differential == 'gaussian':
            self.differential_fn = differential_separation_gaussian
        self.separation_kwargs = kwargs

    def separate(self, mixture: torch.Tensor, num_steps: int = 100):
        device = self.model.device
        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape

        y = sample(
            mixture=mixture,
            denoise_fn=self.model.model.diffusion.denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(self.stems), length_samples).to(device),
            differential_fn=self.differential_fn,
            **self.separation_kwargs,
        )
        return {stem: y[:, i, :] for i, stem in enumerate(self.stems)}

    def separate_with_hint(
            self,
            mixture: torch.Tensor,
            source_with_hint: torch.Tensor,
            mask: torch.Tensor,
            num_steps: int = 100,
    ):
        device = self.model.device
        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape

        y = inpaint(
            source=source_with_hint,
            mask=mask,
            mixture=mixture,
            denoise_fn=self.model.model.diffusion.denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(self.stems), length_samples).to(device),
            differential_fn=self.differential_fn,
            **self.separation_kwargs,
        )

        return {stem: y[:, i:i + 1, :] for i, stem in enumerate(self.stems)}


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
            xs = [x[:, i:i + 1] for i in range(len(stems))]
            xs = [fn(x, sigma=sigma) for fn, x in zip(fns, xs)]
            return torch.cat(xs, dim=1)

        y = sample(
            mixture=mixture,
            denoise_fn=denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(stems), length_samples).to(device),
            differential_fn=differential_separation_gaussian,
            **self.separation_kwargs,
        )
        return {stem: y[:, i:i + 1, :] for i, stem in enumerate(stems)}


class ContextRegularizedSeparator(Separator):
    ...
    # TODO


# Algorithms ------------------------------------------------------------------

@torch.no_grad()
def step(
        x: torch.Tensor,
        i: int,
        denoise_fn: Callable,
        sigmas: torch.Tensor,
        differential_fn: Callable,
        mixture: torch.Tensor = None,
        s_churn: float = 0.0,  # > 0 to add randomness
        use_heun: bool = False,
        **kwargs
):
    sigma, sigma_next = sigmas[i], sigmas[i + 1]

    # Inject randomness
    gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
    sigma_hat = sigma * (gamma + 1)
    x_hat = x + torch.randn_like(x) * (sigma_hat ** 2 - sigma ** 2) ** 0.5

    # Compute conditioned derivative
    d = differential_fn(mixture=mixture, x=x_hat, sigma=sigma_hat, denoise_fn=denoise_fn, **kwargs)

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

    return x


@torch.no_grad()
def sample(
        denoise_fn: Callable,
        sigmas: torch.Tensor,
        noises: Optional[torch.Tensor],
        differential_fn: Callable,
        mixture: torch.Tensor = None,
        s_churn: float = 20.0,  # > 0 to add randomness
        use_heun: bool = False,
        num_resamples: int = 1
):
    # Set initial noise
    x = sigmas[0] * noises  # [batch_size, num-sources, sample-length]

    for i in range(len(sigmas) - 1):
        for r in range(num_resamples):
            x = step(x, i, mixture=mixture, denoise_fn=denoise_fn, sigmas=sigmas, differential_fn=differential_fn,
                     s_churn=s_churn, use_heun=use_heun)
            if r < num_resamples - 1:
                sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                x = x + sigma * torch.randn_like(x)

    return x.cpu().detach()


@torch.no_grad()
def inpaint(
        source: Tensor,
        mask: Tensor,
        denoise_fn: Callable,
        sigmas: Tensor,
        noises: Tensor,
        differential_fn: Callable,
        mixture: Tensor = None,
        num_resamples: int = 1,
        s_churn: float = 20.0,  # > 0 to add randomness
        use_heun: bool = False,
        **kwargs
) -> Tensor:
    x = sigmas[0] * noises

    for i in tqdm(range(len(sigmas) - 1)):
        # Noise source to current noise level

        source_noisy = source + sigmas[i] * torch.randn_like(source)
        for r in range(num_resamples):
            # Merge noisy source and current then denoise
            x = source_noisy * mask + x * mask.logical_not()
            x = step(x, i, mixture=mixture, denoise_fn=denoise_fn, differential_fn=differential_fn,
                     sigmas=sigmas, s_churn=s_churn, use_heun=use_heun, **kwargs)  # type: ignore # noqa
            # Renoise if not last resample step
            if r < num_resamples - 1:
                sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                x = x + sigma * torch.randn_like(x)

    return source * mask + x * mask.logical_not()

