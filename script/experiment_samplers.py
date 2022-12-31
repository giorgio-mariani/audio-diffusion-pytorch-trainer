import itertools
from typing import List, Tuple, Callable
import functools
from numpy.random import Generator, PCG64

import numpy
import torch
import math

import torchaudio
import tqdm
from audio_diffusion_pytorch import KarrasSchedule, ADPM2Sampler
from torch.utils.data import Dataset
from main.scratch_samplers import CrawsonHeunExtractor, CrawsonEulerExtractor
from main.separation import ADPM2Extractor, ADPM2Separator

from main.dataset import SeparationDataset, ChunkedPairsDataset, SeparationSubset
from main.separation import separate, ADPM2Extractor, separate_dataset
from misc import load_model, load_audio
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.resolve().absolute()
DEVICE = torch.device("cuda")
SAMPLE_RATE = 22050


def main():
    model_1 = load_model(ROOT_PATH / "logs/ckpts/2022-11-28-01-14-30/epoch=19999-valid_loss=0.036.ckpt", DEVICE)
    model_2 = load_model(ROOT_PATH / "logs/ckpts/2022-11-22-18-14-22/epoch=10571-valid_loss=0.200.ckpt", DEVICE)
    model_1.to(DEVICE), model_2.to(DEVICE)

    dataset = ChunkedPairsDataset(path_1="/data/MusDB/data/bass/test", path_2="/data/MusDB/data/drums/test",
                                  sample_rate=44100, max_chunk_size=262144, min_chunk_size=262144)
    dataset = TransformDataset(dataset, transform=functools.partial(torch.sum, dim=0, keepdims=True),
                               new_sample_rate=22050)

    indices = numpy.random.default_rng(seed=42).choice(numpy.arange(0, len(dataset), dtype=numpy.int32),
                                                       size=50, replace=False).tolist()

    dataset = SeparationSubset(dataset, indices=indices)

    grid_search = itertools.product(
        [CrawsonEulerExtractor, CrawsonHeunExtractor],
        [1., 20., 40.]
    )

    (ROOT_PATH / "separations").mkdir(exist_ok=True)
    for sampler, s_churn in tqdm.tqdm(grid_search):
        separate_dataset(
            dataset=dataset,
            denoise_fns=[
                model_1.model.diffusion.denoise_fn,
                model_2.model.diffusion.denoise_fn,
            ],
            device=DEVICE,
            separator=sampler,
            save_path=ROOT_PATH / "logs" / "experiments_results" / f"sampler={sampler.__name__}-{s_churn=}",
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=20., rho=7.),
            separation_steps=100,
        )
        print(f"saved separations for sampler={sampler.__name__}-{s_churn=}")

    print("Starting test for ADPM2Extractor")
    grid_search = [ADPM2Extractor, ADPM2Separator]
    for sampler in tqdm.tqdm(grid_search):
        separate_dataset(
            dataset=dataset,
            denoise_fns=[
                model_1.model.diffusion.denoise_fn,
                model_2.model.diffusion.denoise_fn,
            ],
            device=DEVICE,
            separator=sampler,
            save_path=ROOT_PATH / "logs" / "experiments_results" / f"sampler={sampler.__name__}",
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=20., rho=7.),
            separation_steps=100,
        )
        print(f"saved separations for sampler={sampler.__name__}")


class TransformDataset(SeparationDataset):
    def __init__(self, dataset: SeparationDataset, transform: Callable, new_sample_rate: int = None):
        self.new_sample_rate = new_sample_rate if new_sample_rate is not None else dataset.sample_rate
        if new_sample_rate is not None:
            resample_transform = torchaudio.transforms.Resample(
                orig_freq=dataset.sample_rate, new_freq=new_sample_rate
            )
            sequential_transform = lambda x: resample_transform(transform(x))
            self.transform = sequential_transform
        else:
            self.transform = transform
        self.dataset = dataset

    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        return tuple([self.transform(t) for t in self.dataset[item]])

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def sample_rate(self) -> int:
        return self.new_sample_rate


def load_audio_tracks(tracks, start_second: float, length_in_seconds: float):
    length_samples = 2 ** math.ceil(math.log2(length_in_seconds * SAMPLE_RATE))
    start_sample = round(start_second * SAMPLE_RATE)
    end_sample = start_sample + length_samples
    signals = [load_audio(track, SAMPLE_RATE, start_sample, end_sample).squeeze(0) for track in tracks]
    return signals


class DummyDataset(SeparationDataset):
    def __init__(self, tracks: List[torch.Tensor], sample_rate: int, num_samples: int = 12):
        self.tracks = tracks
        self.num_samples = num_samples
        self._sr = sample_rate

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return tuple(self.tracks)

    @property
    def sample_rate(self) -> int:
        return self._sr


if __name__ == "__main__":
    main()