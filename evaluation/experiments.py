import itertools
import json
import math
import functools
from typing import List, Callable, Mapping, Optional, Sequence, Tuple, Union

import torch
import tqdm
from audio_diffusion_pytorch import KarrasSchedule
import numpy
import yaml

from main.module_base import Model
from main.dataset import ChunkedSeparationSubset, ResampleDataset, SeparationDataset, ChunkedSupervisedDataset, SeparationSubset, TransformDataset
from main.separation import IndependentSeparator, ContextualSeparator, separate_dataset, differential_with_dirac, differential_with_gaussian
from script.misc import load_model, load_audio, load_context
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.resolve().absolute()


def load_diffusion_model(path: str, hparams: dict, device: str = "cpu") -> Model:
    model = Model(**{**hparams}).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    return model


def get_subdataset(dataset: SeparationDataset, num_samples: Optional[int] = None, seed:int = 1) -> Tuple[SeparationDataset, Sequence[int]]:
    if num_samples is not None:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), dtype=torch.int, generator=generator)[:num_samples].tolist()
    else:
        indices = list(range(len(dataset)))
    return ChunkedSeparationSubset(dataset, indices), indices


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


def load_musdb_for_eval(musdb_path: Union[Path,str], num_chunks: int) -> SeparationDataset:
    dataset = ChunkedSupervisedDataset(
        stems=["bass", "drums", "other", "vocals"],
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

    
def load_slakh_for_eval(slakh_path: Union[Path,str], num_chunks: int):
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
def main(output_dir: Union[str, Path], use_musdb: bool = True):
    output_dir = Path(output_dir)
    device = torch.device("cuda:0")

    if not use_musdb:
        dataset = load_slakh_for_eval("/home/giorgio_mariani/audio-diffusion-pytorch-trainer/data/Slakh/test", num_chunks=30)
        irene_ckpt_path = Path("/home/giorgio_mariani/Documents/audio-diffusion-pytorch-trainer/logs/ckpts/")
        dataset_name = "Slakh"

        model_bass = load_model(ROOT_PATH / "logs/ckpts/logical-butterfly-181_epoch=11447_loss=0.005.ckpt", device)
        model_guitar = load_model(ROOT_PATH / "logs/ckpts/radiant-wind-181_epoch=4666_loss=0.014.ckpt", device)
        model_drums = load_model(irene_ckpt_path / "drums_slack.ckpt", device)
        model_piano = load_model(irene_ckpt_path / "piano_slack.ckpt", device)
        stem_to_model={
                "bass": model_bass, 
                "drums": model_drums, 
                "guitar": model_guitar, 
                "piano": model_piano,
            }

    else:
        dataset = load_musdb_for_eval("/home/giorgio_mariani/Documents/audio-diffusion-pytorch-trainer/data/MusDB/test", num_chunks=30)
        dataset_name = "MusDB"
        ckpts_path = ROOT_PATH / "data"
        model_bass = load_model(ckpts_path / "rustful-dust-117_epoch=19999-loss=0.026.ckpt", device)
        model_vocals = load_model(ckpts_path / "royal-breeze-162_epoch=29142_loss=0.138.ckpt", device)
        model_drums = load_model(ckpts_path / "usual-frost-113_epoch=10571-loss=0.200.ckpt", device)
        model_other = load_model(ckpts_path / "copper-wood-169_epoch=28571-valid_loss=0.206.ckpt", device)
        stem_to_model={
                "bass": model_bass, 
                "drums": model_drums, 
                "other": model_other, 
                "vocals": model_vocals,
            }
    
    hparams = {
        "sigma_min": [1e-4],
        "sigma_max": [20.0],
        "rho": [7],
        "s_churn": [40.0],
        "num_steps": [150],
        "use_heun": [False],
    }
    
    sep_factory = functools.partial(
        separator_factory,
        separator=functools.partial(IndependentSeparator, stem_to_model)
    )
    
    hparams_search(
        dataset, 
        sep_factory, 
        hparams={"likelihood": ["dirac"], "source_id": [0,1,2,3],**hparams}, 
        save_path=output_dir / f"weak_{dataset_name}_dirac_22050_source",
    )

    hparams_search(
        dataset, 
        sep_factory, 
        hparams={"likelihood": ["gaussian"], "gamma_coeff": [0.25, 0.5, 0.75, 1.0], **hparams}, 
        save_path=output_dir / f"weak_{dataset_name}_gaussian_22050_gamma",
    )


@torch.no_grad()
def weakly_slakh_4stems(
    output_dir: Union[str, Path],
    num_samples: int = None,
    num_resamples: int = 1,
    num_steps: int = 150,
    batch_size: int = 16,
    resume: bool = True,
    device: float = torch.device("cuda:0"),
    s_churn: float = 20.0,
    source_id: int = -1,
    ):
    output_dir = Path(output_dir)

    dataset = ChunkedSupervisedDataset(
        audio_dir="/data/Slakh_supervised/test",
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=44100,
        max_chunk_size=262144 * 2,
        min_chunk_size=262144 * 2,
    )

    model_bass = load_model(ROOT_PATH / "ckpts/laced-dream-329-(SLAKH_BASS_v2)-epoch=443.ckpt", device)
    model_guitar = load_model(ROOT_PATH / "ckpts/honest-fog-332_(SLAKH_GUITAR_v2)-epoch=407.ckpt", device)
    model_drums = load_model(ROOT_PATH / "ckpts/ancient-voice-289-(SLAKH_DRUMS_v2)-epoch=258.ckpt", device)
    model_piano = load_model(ROOT_PATH / "ckpts/ruby-dew-290-(SLAKH_PIANO_v2)-epoch=236.ckpt", device)

    separator = IndependentSeparator(
        stem_to_model={
            "bass": model_bass,
            "drums": model_drums,
            "guitar": model_guitar,
            "piano": model_piano,
        },
        sigma_schedule=KarrasSchedule(sigma_min=1e-4, sigma_max=1.0, rho=7.0),
        differential_fn=functools.partial(differential_with_dirac, source_id=source_id),
        s_churn=s_churn,
        num_resamples=num_resamples,
    )

    context_4stems(
        output_dir=output_dir,
        dataset=dataset,
        separator=separator,
        sample_rate=22050,
        num_samples=num_samples,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume,
    )



@torch.no_grad()
def context_slakh_4stems(
        output_dir: Union[str, Path],
        num_samples: int = -1,
        s_churn: float = 20.,
        num_resamples: int = 1,
        source_id: int = 0,
        gradient_mean: bool = False,
        num_steps: int = 150,
        batch_size: int = 16,
        num_separations: int = 1,
        num_gibbs_steps: int = 1,
        hint_fixed_sources_idx: List[int] = [],
        use_heun=False,
        resume = False,
        sigma_min=1e-4,
        gibbs_sources=None
    
    ):
    output_dir = Path(output_dir)
    device = torch.device("cuda:0")
    sigma_min, sigma_max = sigma_min, 1.0
    
    assert gibbs_sources is None or len(gibbs_sources)==(num_gibbs_steps-1), "len(gibbs_sources) è diverso da num_gibbs_steps-1"
    
    dataset = ChunkedSupervisedDataset(
        audio_dir="/home/giorgio_mariani/Documents/audio-diffusion-pytorch-trainer/data/Slakh_track_first/test",
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=44100,
        max_chunk_size=262144 * 2,
        min_chunk_size=262144 * 2,
    )
    #print("len(dataset)", len(dataset))
    if num_samples != -1:
        print("num_samples",num_samples)
        generator = torch.Generator().manual_seed(1)
        #indices = torch.randint(high=len(dataset), size=(num_samples,), dtype=torch.int, generator=generator).tolist()
        indices = torch.randperm(len(dataset), dtype=torch.int, generator=generator)[:num_samples].tolist()
    else:
        indices = list(range(len(dataset)))
        
    dataset = ChunkedSeparationSubset(dataset, indices=indices)
    resampled_dataset = ResampleDataset(dataset=dataset, new_sample_rate=22050)
    ckpts_path = Path("/home/giorgio_mariani/Documents/audio-diffusion-pytorch-trainer/logs/ckpts")
    model_cpu = load_context(ckpts_path / "avid-darkness-164_epoch=419-valid_loss=0.015.ckpt", "cpu", 4)
    model = model_cpu.to(device)
    del model_cpu

    separator = ContextualSeparator(
        model=model,
        stems=["bass", "drums", "guitar", "piano"],
        sigma_schedule=KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0),
        differential_fn=differential_with_dirac,
        s_churn=s_churn,
        num_resamples=num_resamples,
        source_id=source_id,
        gradient_mean=gradient_mean,
        use_heun=use_heun
    )

    chunk_data = []
    #print("len(dataset)",len(dataset))
    #print("len(indices)",len(indices))
    print("indices",indices)
        
    for i in range(len(indices)):
        # print("i",i)
        start_sample, end_sample = dataset.get_chunk_indices(indices[i])
        chunk_data.append(
            {
                "chunk_index": i,
                "track": dataset.get_chunk_track(indices[i]),
                "start_chunk_sample": start_sample,
                "end_chunk_sample": end_sample,
                "start_chunk_seconds": start_sample / 22050,
                "end_chunk_in_seconds": end_sample / 22050,
            }
        )
        
    
    if not resume:
        output_dir.mkdir(parents=True)
        
    with open(output_dir / "chunk_data.json", "w") as f:
        json.dump(chunk_data, f)
    
    for n in range(num_separations):
        separate_dataset(
            dataset=resampled_dataset,
            separator=separator,
            save_path=output_dir / f"sep_round_{n}",
            num_steps=num_steps,
            hint_fixed_sources_idx=hint_fixed_sources_idx,
            batch_size=batch_size,
            num_gibbs_steps=num_gibbs_steps,
            resume=resume,
            gibbs_sources=gibbs_sources
        )


@torch.no_grad()
def context_4stems(
        output_dir: Union[str, Path],
        dataset: SeparationDataset,
        separator: ContextualSeparator,
        sample_rate: int,
        num_samples: Optional[int] = None,
        num_steps: int = 150,
        batch_size: int = 16,
        num_separations: int = 1,
        num_gibbs_steps: int = 1,
        hint_fixed_sources_idx: Sequence[int] = tuple([]),
        resume: bool = False,
        transform_fn: Callable = None,
    ):

    resampled_dataset, indices = get_subdataset(dataset, num_samples)
    resampled_dataset = ResampleDataset(dataset=resampled_dataset, new_sample_rate=sample_rate)
    if transform_fn is not None:
        resampled_dataset = TransformDataset(resampled_dataset, transform=transform_fn)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    chunk_data = []
    for i in range(len(indices)):
        start_sample, end_sample = dataset.get_chunk_indices(indices[i])
        chunk_data.append(
            {
                "chunk_index": i,
                "track": dataset.get_chunk_track(indices[i]),
                "start_chunk_sample": start_sample,
                "end_chunk_sample": end_sample,
                "start_chunk_seconds": start_sample / dataset.sample_rate,
                "end_chunk_in_seconds": end_sample / dataset.sample_rate,
            }
        )

    with open(output_dir / "chunk_data.json", "w") as f:
        json.dump(chunk_data, f)

    for n in range(num_separations):
        separate_dataset(
            dataset=resampled_dataset,
            separator=separator,
            save_path=output_dir / f"sep_round_{n}",
            num_steps=num_steps,
            hint_fixed_sources_idx=hint_fixed_sources_idx,
            batch_size=batch_size,
            num_gibbs_steps=num_gibbs_steps,
            resume=resume
        )


def context_musdb_4stems(
    musdb18_path: Path,
    ckpt_path: Path,
    output_dir: Path,
    sample_rate: int,
    device: str = "cuda:0",
    num_samples: Optional[int] = None,
    sigma_min: float = 1e-4,
    sigma_max: float = 1.0,
    s_churn: float = 20.0,
    source_id: int = -1,
    num_steps: int = 150,
    batch_size: int = 16,
    resume: bool = False
    ):

    dataset = ChunkedSupervisedDataset(
        audio_dir=musdb18_path,
        stems=["bass", "drums", "other", "vocals"],
        sample_rate=44100,
        max_chunk_size=int(262144 * 44100 / sample_rate),
        min_chunk_size=int(262144 * 44100 / sample_rate),
    )

    separator = ContextualSeparator(
        model=load_diffusion_model(ckpt_path, yaml.safe_load(ckpt_path.with_suffix(".yaml").read_text()), device),
        stems=["bass", "drums", "guitar", "piano"],
        sigma_schedule=KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0),
        differential_fn=differential_with_dirac,
        s_churn=s_churn,
        num_resamples=2,
        source_id=source_id,
        gradient_mean=False,
    )

    context_4stems(
        output_dir=output_dir,
        dataset=dataset,
        separator=separator,
        sample_rate=sample_rate,
        num_samples=num_samples,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume,
        transform_fn=functools.partial(torch.mean, dim=0, keepdims=True)
    )


if __name__ == "__main__":
    # source_id = -1 changes the source at each separation step
    # nota, se trova gia output_dir non la sovrascrive, devi cancellarla a mano (ed è giusto così ahaha)
    # Se num_samples = -1 separa tutto il dataset
    # resume serve a far ripartire la separazione da dove si è interrotta, se si è interrotta per sbaglio
    num_steps=150
    source_id=0 
    num_resamples=1
    num_separations=1
    num_gibbs_steps=10
    hint_fixed_sources_idx=[]
    use_heun=False
    sigma_min = 1e-4
    gibbs_sources=None
    s_churn=20.

    context_slakh_4stems(output_dir=f"separations/context_slakh_last_{num_steps=}_{source_id=}_{num_resamples=}_{num_separations=}_{num_gibbs_steps=}_{hint_fixed_sources_idx=}_{use_heun=}_{gibbs_sources=}_{s_churn=}", 
                         num_samples=-1, num_steps=num_steps, batch_size=128, source_id=source_id, 
                         gradient_mean=False, num_resamples=num_resamples, 
                         s_churn=s_churn, num_separations=num_separations, num_gibbs_steps=num_gibbs_steps, gibbs_sources=gibbs_sources,
                         hint_fixed_sources_idx=hint_fixed_sources_idx, resume=False, use_heun=use_heun, sigma_min= sigma_min)
