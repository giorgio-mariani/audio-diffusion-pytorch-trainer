import itertools
from typing import List

import torch
import math

import tqdm
from audio_diffusion_pytorch import KarrasSchedule
from torch.utils.data import Dataset

from main.dataset import SeparationDataset
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

    s1, s2 = load_audio_tracks(
        [
            "/data/MusDB/data/test_MusDB_reduced/Al James - Schoolboy Facination/bass.wav",
            "/data/MusDB/data/test_MusDB_reduced/Al James - Schoolboy Facination/drums.wav",
        ],
        start_second=170,
        length_in_seconds=10,
    )

    # convert to mono
    s1, s2 = s1.sum(dim=0, keepdims=True), s2.sum(dim=0, keepdims=True)

    grid_search = itertools.product(
        [1e-4, 5e-2, 0.1],
        [1.0, 5.0, 20.0],
        [5.0, 7.0, 9.0],
        [100]
    )

    grid_search = itertools.product(
        [1e-4],
        [20.0],
        [7.0],
        [25, 50, 75, 100, 250, 500]
    )

    dataset = DummyDataset([s1, s2], num_samples=20, sample_rate=SAMPLE_RATE)
    (ROOT_PATH / "separations").mkdir(exist_ok=True)
    for smin, smax, rho, num_steps in tqdm.tqdm(grid_search):
        separate_dataset(
            dataset=dataset,
            denoise_fns=[
                model_1.model.diffusion.denoise_fn,
                model_2.model.diffusion.denoise_fn,
            ],
            device=DEVICE,
            separator=ADPM2Extractor,
            save_path=ROOT_PATH / "separations" / f"{smin}-{smax}-{rho}-{num_steps}",
            sigma_schedule=KarrasSchedule(sigma_min=smin, sigma_max=smax, rho=rho),
            separation_steps=num_steps,
        )
        print(f"saved separations {smin}-{smax}-{rho}")


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