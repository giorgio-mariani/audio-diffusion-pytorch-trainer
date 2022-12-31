import abc
import itertools
from abc import ABC
import functools
import math
import warnings
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, List

import random

import librosa
import torch
import torch.nn as nn
import torchaudio
from audio_data_pytorch import AllTransform
from audio_data_pytorch.datasets.wav_dataset import get_all_wav_filenames
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import Dataset


class ChunkedWAVDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        max_chunk_size: int,
        min_chunk_size: int = 1,
        recursive: bool = True,
        transforms: Optional[Callable] = None,
    ):
        super().__init__()

        # Load list of files
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.transforms = transforms

        self.index_to_file, self.index_to_chunk = [], []
        for file in self.wavs:
            t1 = self.get_track(file)
            available_chunks = get_chunks(t1, max_chunk_size, min_chunk_size)
            self.index_to_file.extend([file] * available_chunks)
            self.index_to_chunk.extend(range(available_chunks))

        assert len(self.index_to_chunk) == len(self.index_to_file)

    @functools.lru_cache(1024)
    def get_track(self, filename: str) -> Tensor:
        assert filename in self.wavs
        waveform, sample_rate = torchaudio.load(filename)
        return waveform

    def __len__(self):
        return len(self.index_to_file)

    def get_chunk_track(self, item: int) -> str:
        return self.index_to_file[item]

    def get_chunk_indices(self, item: int) -> Tuple[int, int]:
        ci = self.index_to_chunk[item]
        return ci * self.max_chunk_size, (ci + 1) * self.max_chunk_size

    def __getitem__(self, item: int) -> Tensor:
        chunk_start, chunk_stop = self.get_chunk_indices(item)
        waveform = self.get_track(self.get_chunk_track(item))
        waveform = waveform[:, chunk_start:chunk_stop]

        # pad sequence if necessary
        if waveform.shape[-1] < self.max_chunk_size:
            pad_shape = list(waveform.shape)
            pad_shape[-1] = self.max_chunk_size
            padded_waveform = torch.zeros(pad_shape, dtype=waveform.dtype, device=waveform.device)
            padded_waveform[:,:waveform.shape[-1]] = waveform
            waveform = padded_waveform

        if self.transforms is not None:
            waveform = self.transforms(waveform)
        return waveform


class SeparationDataset(Dataset, ABC):
    @abc.abstractmethod
    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        ...


class SeparationSubset(SeparationDataset):
    def __init__(self, dataset: SeparationDataset, indices: List[int]):
        self.dataset = dataset
        self.subset = torch.utils.data.Subset(dataset, indices)
        self.indices = indices

    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        return self.subset[item]

    def __len__(self) -> int:
        return len(self.subset)

    @property
    def sample_rate(self) -> int:
        return self.dataset.sample_rate


class UnionSeparationDataset(SeparationDataset):
    def __init__(self, datasets: List[Dataset], sample_rate: int):
        self.datasets = datasets
        self.sr = sample_rate

    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        tracks = tuple(dataset[item] for dataset in self.datasets)
        # TODO: chack same shape, dtype, device
        # TODO: check is audio
        return tracks

    def __len__(self) -> int:
        return min([len(data) for data in self.datasets])

    @property
    def sample_rate(self) -> int:
        return self.sr


class TrackPairsDataset(SeparationDataset):
    def __init__(
        self,
        instrument_1_audio_dir: Union[str, Path],
        instrument_2_audio_dir: Union[str, Path],
        sample_rate: int,
        sample_eps: int = 2000
    ):
        super().__init__()
        self.sr = sample_rate
        self.sample_eps = sample_eps

        # Load list of files and starts/durations
        self.dir_1 = Path(instrument_1_audio_dir)
        self.dir_2 = Path(instrument_2_audio_dir)
        dir_1_files = librosa.util.find_files(str(self.dir_1))
        dir_2_files = librosa.util.find_files(str(self.dir_2))

        # get filenames
        dir_1_files = set(sorted([Path(f).name for f in dir_1_files]))
        dir_2_files = set(sorted([Path(f).name for f in dir_2_files]))
        self.filenames = list(sorted(dir_1_files.intersection(dir_2_files)))

        if len(self.filenames) != len(dir_1_files):
            unused_tracks = len(dir_1_files.difference(self.filenames))
            warnings.warn(f"Not using all available tracks in {self.dir_1} ({unused_tracks})")

        if len(self.filenames) != len(dir_2_files):
            unused_tracks = len(dir_2_files.difference(self.filenames))
            warnings.warn(f"Not using all available tracks in {self.dir_2} ({unused_tracks})")

    def __len__(self):
        return len(self.filenames)

    def get_tracks(self, filename: str) -> Tuple[torch.Tensor, ...]:
        assert filename in self.filenames
        tracks = load_audio_tracks(
            paths=[self.dir_1 / filename, self.dir_2 / filename],
            sample_rate=self.sr,
        )

        channels, samples = zip(*[t.shape for t in tracks])
        #for c in channels:
        #    assert c == 1

        for s1, s2 in itertools.product(samples, samples):
            assert abs(s1 - s2) <= self.sample_eps, f"{filename}: {abs(s2 - s2)}"
            if s1 != s1:
                warnings.warn(
                    f"The tracks with name {filename} have a different number of samples ({s1}, {s2})"
                )

        n_samples = min(samples)
        return tuple(t[:, :n_samples] for t in tracks)

    @property
    def sample_rate(self) -> int:
        return self.sr

    def __getitem__(self, item):
        return self.get_tracks(self.filenames[item])


class ChunkedPairsDataset(TrackPairsDataset):
    def __init__(
        self,
        path_1: Union[str, Path],
        path_2: Union[str, Path],
        sample_rate: int,
        max_chunk_size: int,
        min_chunk_size: int,
    ):
        # Load list of files and starts/durations
        super().__init__(path_1, path_2, sample_rate)

        self.max_chunk_size = max_chunk_size
        self.available_chunk = {}
        self.index_to_file, self.index_to_chunk = [], []

        for file in self.filenames:
            t1, t2 = self.get_tracks(file)
            available_chunks = get_nonsilent_chunks(t1 + t2, max_chunk_size, min_chunk_size)
            self.available_chunk[file] = available_chunks
            self.index_to_file.extend([file] * len(available_chunks))
            self.index_to_chunk.extend(available_chunks)

        assert len(self.index_to_chunk) == len(self.index_to_file)

    @functools.lru_cache(1024)
    def load_tracks(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_tracks(filename)

    def __len__(self):
        return len(self.index_to_file)

    def get_chunk_track(self, item: int) -> str:
        return self.index_to_file[item]

    def get_chunk_indices(self, item: int) -> Tuple[int, int]:
        ci = self.index_to_chunk[item]
        return ci * self.max_chunk_size, (ci + 1) * self.max_chunk_size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        chunk_start, chunk_stop = self.get_chunk_indices(item)
        t1, t2 = self.load_tracks(self.get_chunk_track(item))
        t1, t2 = t1[:, chunk_start:chunk_stop], t2[:, chunk_start:chunk_stop]
        return t1, t2


def load_audio_tracks(paths: List[Union[str, Path]], sample_rate: int) -> Tuple[torch.Tensor, ...]:
    signals, sample_rates = zip(*[torchaudio.load(path) for path in paths])
    for sr in sample_rates:
        assert sr == sample_rate, f"sample rate {sr} is different from target sample rate {sample_rate}"
    return tuple(signals)


def assert_is_audio(*signal: torch.Tensor):
    for s in signal:
        assert len(s.shape) == 2
        assert s.shape[0] == 1 or s.shape[0] == 2


def is_silent(signal: torch.Tensor, silence_threshold: float = 1.5e-5) -> bool:
    assert_is_audio(signal)
    num_samples = signal.shape[-1]
    return torch.linalg.norm(signal) / num_samples < silence_threshold


def get_nonsilent_chunks(
    track: torch.Tensor,
    max_chunk_size: int,
    min_chunk_size: int = 0,
):
    assert_is_audio(track)
    _, num_samples = track.shape
    num_chunks = num_samples // max_chunk_size + int(num_samples % max_chunk_size != 0)

    available_chunks = []
    for i in range(num_chunks):
        chunk = track[:, i * max_chunk_size : (i + 1) * max_chunk_size]
        _, chunk_samples = chunk.shape

        if not is_silent(chunk) and chunk_samples >= min_chunk_size:
            available_chunks.append(i)

    return available_chunks


def get_chunks(track: Tensor, max_chunk_size: int, min_chunk_size: int):
    assert min_chunk_size > 0
    assert max_chunk_size > min_chunk_size

    _, num_samples = track.shape

    # get number of chunks mith max length
    num_chunks = num_samples // max_chunk_size

    # check if last chunk has enough samples
    if num_samples - max_chunk_size * (num_samples // max_chunk_size) >= min_chunk_size:
        num_chunks = num_chunks + 1
    return num_chunks


# Transforms ===========================================================================================================
class FlipSign(nn.Module):
    """
    Random sign flip.
    """
    def forward(self, wav):
        if self.training:
            signs = torch.randint(2, (1,1), device=wav.device, dtype=torch.float32)
            wav = wav * (2 * signs - 1)
        return wav


class Scale(nn.Module):
    def __init__(self, scales_sampler: Distribution, proba=1.):
        super().__init__()
        self.proba = proba
        self.scales_sampler = scales_sampler

    def forward(self, wav):
        if self.training and random.random() < self.proba:
            wav[:, :] = wav[:, :] * self.scales_sampler.sample()
        return wav


class TimeStretch(nn.Module):
    def __init__(
            self,
            timerates_sampler: Distribution,
            proba: float = 1.0,
            hop_length: Optional[int] = None,
            n_fft: Optional[int] = None,
    ):
        super().__init__()
        self.proba = proba
        self.timerates_sampler = timerates_sampler
        self.spec = torchaudio.transforms.Spectrogram(hop_length=hop_length, n_fft=n_fft, power=None)
        self.invspec = torchaudio.transforms.InverseSpectrogram(hop_length=hop_length, n_fft=n_fft)
        self.time_stretcher = torchaudio.transforms.TimeStretch(hop_length, n_freq=n_fft // 2 + 1)

    def forward(self, wav):
        if self.training and random.random() < self.proba:
            spec = self.spec(wav)
            spec_ts = self.time_stretcher(spec, overriding_rate=self.timerates_sampler.sample())
            wav = self.invspec(spec_ts)
        return wav


TimeShift = TimeStretch


class PitchShift(nn.Module):
    def __init__(
            self,
            sample_rate: int,
            pitchshifts_sampler: Distribution,
            proba: float = 1.0,
            hop_length: Optional[int] = None,
            n_fft: Optional[int] = None,
    ):
        super().__init__()
        self.proba = proba
        self.pitchshifts_sampler = pitchshifts_sampler
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sr = sample_rate

    def forward(self, wav):
        if self.training and random.random() < self.proba:
            wav = torchaudio.functional.pitch_shift(
                waveform=wav,
                sample_rate=self.sr,
                n_steps=round(self.pitchshifts_sampler.sample().item()),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
        return wav


def get_random_sample(
    dataset: SeparationDataset,
    num_samples: Optional[int] = None,
    seed: int = 0,
) -> SeparationSubset:
    data_length = len(dataset)
    num_samples = num_samples if num_samples is not None else data_length
    generator = torch.Generator()
    generator.manual_seed(seed)
    samples = torch.randperm(data_length, generator=generator).tolist()
    return SeparationSubset(dataset, samples[: num_samples])
