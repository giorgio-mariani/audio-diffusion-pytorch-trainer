import itertools
import json
import math
import functools
from typing import List, Callable, Mapping, Union

import torch
import tqdm
from audio_diffusion_pytorch import KarrasSchedule
from torch.utils.data import Dataset
import numpy
import numpy as np

from main.dataset import ResampleDataset, SeparationDataset, ChunkedSupervisedDataset, SeparationSubset, TransformDataset
from main.separation import IndependentSeparator, ContextualSeparator, separate_dataset, differential_with_dirac, differential_with_gaussian
from script.misc import load_model, load_audio, load_context
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.resolve().absolute()
DEVICE = torch.device("cuda")
SAMPLE_RATE = 22050


def hparams_search(
    dataset: SeparationDataset,
    separator_factory: Callable,
    hparams: Mapping[str, list],
    save_path: Path,
):
    hparams_names = hparams.keys()
    grid_search = itertools.product(*hparams.values())
    
    save_path.mkdir(exist_ok=True)
    for i, hparams_values in enumerate(tqdm.tqdm(grid_search)):
        
        sep_kwargs = {k:v for k,v in zip(hparams_names, hparams_values)}
        separator = separator_factory(**sep_kwargs)
        
        # log hyper parameters
        with open(save_path / f"experiment-{i}-hparams.json", "w") as f:
            json.dump(sep_kwargs, f)
        
        # separate data
        print(dataset)
        print(len(dataset))
        separate_dataset(
            dataset=dataset,
            separator=separator,
            save_path=save_path / f"experiment-{i}",
            num_steps=sep_kwargs.pop("num_steps", 100)
        )

        print(f"saved separations {i}")


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


def load_musdb_for_eval(musdb_path: Path, num_chunks: int) -> SeparationDataset:
    dataset = ChunkedSupervisedDataset(
        stems=[],
        audio_dir=musdb_path,
        sample_rate=44100, 
        max_chunk_size=262144 * 2, 
        min_chunk_size=262144 * 2,
    )
    
    dataset = TransformDataset(
        dataset, 
        transform=functools.partial(torch.sum, dim=0, keepdims=True),
    )

    dataset = ResampleDataset(dataset=dataset, new_sample_rate=22050)

    rng = numpy.random.default_rng(seed=42)
    indices = rng.choice(
        numpy.arange(0, len(dataset), dtype=numpy.int32), size=num_chunks, replace=False
    ).tolist()

    return SeparationSubset(dataset, indices=indices)

    
def load_slakh_for_eval(slakh_path: Path, num_chunks: int):
    dataset = ChunkedSupervisedDataset(
        audio_dir=slakh_path,
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=44100,
        max_chunk_size=262144*2,
        min_chunk_size=262144*2,
    )

    dataset = ResampleDataset(dataset=dataset, new_sample_rate=22050)
    
    rng = numpy.random.default_rng(seed=42)
    indices = rng.choice(
        numpy.arange(0, len(dataset), dtype=numpy.int32), size=num_chunks, replace=False
    ).tolist()

    return SeparationSubset(dataset, indices=indices)

    
def separator_factory(separator, **kwargs):
    likelihood_type = kwargs.pop("likelihood", "dirac")
    gamma_coeff = kwargs.pop("gamma_coeff", 1.0)
    source_id = kwargs.pop("source_id", 0)
    
    if likelihood_type == "dirac":
        differential_fn = functools.partial(
            differential_with_dirac,
            source_id=source_id,
        )
    elif likelihood_type == "gaussian":
        differential_fn = functools.partial(
            differential_with_gaussian,
            gamma_fn=lambda x: gamma_coeff * x,
        )
    else:
        assert False
    
    smin = kwargs.pop("sigma_min", 1e-4)
    smax = kwargs.pop("sigma_max", 1.0)
    rho = kwargs.pop("rho", 7.0)
    num_steps = kwargs.pop("num_steps", 100)
    return separator(
        sigma_schedule=KarrasSchedule(sigma_min=smin, sigma_max=smax, rho=rho),
        differential_fn=differential_fn,
        **kwargs,
    )
    
@torch.no_grad()
def main(output_dir: Union[str, Path]):
    output_dir = Path(output_dir)
    device = torch.device("cuda:0")
    dataset = load_slakh_for_eval("data/slakh_supervised/test", num_chunks=30)
    #irene_ckpt_path = Path("/home/irene/Documents/audio-diffusion-pytorch-trainer/logs/ckpts/")
    
    #model_bass = load_model(ROOT_PATH / "logs/ckpts/logical-butterfly-181_epoch=11447_loss=0.005.ckpt", device)
    #model_guitar = load_model(ROOT_PATH / "logs/ckpts/radiant-wind-181_epoch=4666_loss=0.014.ckpt", device)
    #model_drums = load_model(irene_ckpt_path / "drums_slack.ckpt", device)
    #model_piano = load_model(irene_ckpt_path / "piano_slack.ckpt", device)
    model_context = load_context(ROOT_PATH / "logs/ckpts/all_slakh_epoch=419.ckpt", device, 4)
    
    hparams = {
        "sigma_min": [1e-4],
        "sigma_max": [1.0],
        "rho": [7],
        "s_churn": [20.0],
        "num_steps": [150],
    }
    
    sep_factory = functools.partial(
        separator_factory,
        separator=functools.partial(
            ContextualSeparator,
            #stem_to_model={
            #    "bass": model_bass,
            #    "drums": model_drums,
            #    "guitar": model_guitar,
            #    "piano": model_piano,
            #}
            stems=["bass", "drums", "guitar", "piano"],
            model=model_context
        )
    )
    
    #hparams_search(
    #    dataset,
    #    sep_factory,
    #    hparams={"likelihood": ["dirac"], "source_id": [0,1,2,3],**hparams},
    #    save_path=output_dir / "context_slakh_dirac_22050_source",
    #)

    hparams_search(
        dataset,
        sep_factory,
        hparams={"likelihood": ["gaussian"], "gamma_coeff": [0.06, 0.125, 0.3725], **hparams},
        save_path=output_dir / "context_slakh_gaussian_22050_gamma_2",
    )


@torch.no_grad()
def weakly_slakh(output_dir: Union[str, Path]):
    output_dir = Path(output_dir)
    device = torch.device("cuda:0")

    dataset = ChunkedSupervisedDataset(
        audio_dir="/data/Slakh_supervised/test",
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=44100,
        max_chunk_size=262144 * 2,
        min_chunk_size=262144 * 2,
    )

    resampled_dataset = ResampleDataset(dataset=dataset, new_sample_rate=22050)

    model_bass = load_model("/data/ckpts/bass_epoch=14236.ckpt", device)
    model_guitar = load_model("/data/ckpts/guitar_epoch=6098.ckpt", device)
    model_drums = load_model("/data/ckpts/drums_epoch=933.ckpt", device)
    model_piano = load_model("/data/ckpts/piano_epoch=880.ckpt", device)

    separator = IndependentSeparator(
        stem_to_model={
            "bass": model_bass,
            "drums": model_drums,
            "guitar": model_guitar,
            "piano": model_piano,
        },
        sigma_schedule=KarrasSchedule(sigma_min=1e-4, sigma_max=1.0, rho=7.0),
        s_churn=40.0,
    )

    chunk_data = []
    for i in range(len(dataset)):
        start_sample, end_sample = dataset.get_chunk_indices(i)
        chunk_data.append(
            {
                "chunk_index": i,
                "track": dataset.get_chunk_track(i),
                "start_chunk_sample": start_sample,
                "end_chunk_sample": end_sample,
                "start_chunk_seconds": start_sample / 44100,
                "end_chunk_in_seconds": end_sample / 44100,
            }
        )

    separate_dataset(
        dataset=resampled_dataset,
        separator=separator,
        save_path=output_dir,
        num_steps=200,
    )

    with open(output_dir / "chunk_data.json", "w") as f:
        json.dump(chunk_data, f)


if __name__ == "__main__":
    weakly_slakh("separation/weakly_dirac_all_slakh")