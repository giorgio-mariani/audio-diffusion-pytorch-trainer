import abc
import itertools
from abc import ABC
import functools
import math
import warnings
from pathlib import Path
import os
from typing import Union, Optional, Callable, Tuple, Sequence, List

import random

import librosa
import torch
import torch.nn as nn
import torchaudio
from audio_data_pytorch import AllTransform
from audio_data_pytorch.datasets.wav_dataset import get_all_wav_filenames, WAVDataset
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import Dataset


class CachedWAVDataset(WAVDataset):
    def __init__(
            self,
            path: Union[str, Sequence[str]],
            recursive: bool = False,
            transforms: Optional[Callable] = None,
            sample_rate: Optional[int] = None,
    ):
        super().__init__(path, recursive, transforms, sample_rate)

    @functools.lru_cache(1024)
    def load_audio(self, wav_file: str) -> Tuple[Tensor, int]:
        return torchaudio.load(wav_file)

    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        waveform, sample_rate = self.load_audio(self.wavs[idx])

        if self.sample_rate and sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )(waveform)

        if self.transforms:
            waveform = self.transforms(waveform)

        return waveform

    def __len__(self) -> int:
        return len(self.wavs)

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

        self.index_to_file, self.index_to_track_chunk = [], []
        for file in self.wavs:
            t1 = self.get_track(file)
            available_chunks = get_chunks(t1, max_chunk_size, min_chunk_size)
            self.index_to_file.extend([file] * available_chunks)
            self.index_to_track_chunk.extend(range(available_chunks))

        assert len(self.index_to_track_chunk) == len(self.index_to_file)

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
        ci = self.index_to_track_chunk[item]
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


class TransformDataset(SeparationDataset):
    def __init__(self, dataset: SeparationDataset, transform: Callable):
        #self.new_sample_rate = new_sample_rate if new_sample_rate is not None else dataset.sample_rate
        #if new_sample_rate is not None:
        #    resample_transform = torchaudio.transforms.Resample(
        #        orig_freq=dataset.sample_rate,
        #        new_freq=new_sample_rate,
        #    )
        #    sequential_transform = lambda x: resample_transform(transform(x))
        #    self.transform = sequential_transform
        #else:
        #    self.transform = transform
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        return tuple([self.transform(t) for t in self.dataset[item]])

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def sample_rate(self) -> int:
        return self.dataset.sample_rate


class ResampleDataset(SeparationDataset):
    def __init__(self, dataset: SeparationDataset, new_sample_rate: int):
        self.new_sample_rate = new_sample_rate if new_sample_rate is not None else dataset.sample_rate
        self.transform = torchaudio.transforms.Resample(
            orig_freq=dataset.sample_rate,
            new_freq=new_sample_rate,
        )
        self.dataset = dataset

    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        return tuple([self.transform(t) for t in self.dataset[item]])

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def sample_rate(self) -> int:
        return self.new_sample_rate
    


class SupervisedDataset(SeparationDataset):
    def __init__(
        self,
        audio_dir: Union[str, Path],
        stems: List[str],
        sample_rate: int,
        sample_eps_in_sec: int = 0.1
    ):
        super().__init__()
        self.sr = sample_rate
        self.sample_eps = round(sample_eps_in_sec * sample_rate)

        # Load list of files and starts/durations
        self.audio_dir = Path(audio_dir)
        self.tracks = sorted(os.listdir(self.audio_dir))
        self.stems = stems
        
        #TODO: add check if stem is never present in any track

    def __len__(self):
        return len(self.filenames)

    #@functools.lru_cache(256)
    def get_tracks(self, track: str) -> Tuple[torch.Tensor, ...]:
        assert track in self.tracks
        stem_paths = {stem: self.audio_dir / track / f"{stem}.wav" for stem in self.stems}
        stem_paths = {stem: stem_path for stem, stem_path in stem_paths.items() if stem_path.exists()}
        assert len(stem_paths) >= 1, track
        
        stems_tracks = {}
        for stem, stem_path in stem_paths.items():
            audio_track, sr = load_audio_track(path=stem_path)
            assert sr == self.sample_rate, f"sample rate {sr} is different from target sample rate {self.sample_rate}"
            stems_tracks[stem] = audio_track
                        
        channels, samples = zip(*[t.shape for t in stems_tracks.values()])
        
        #TODO add assert on channels

        for s1, s2 in itertools.product(samples, samples):
            assert abs(s1 - s2) <= self.sample_eps, f"{track}: {abs(s1 - s2)}"
            if s1 != s2:
                warnings.warn(
                    f"The tracks with name {track} have a different number of samples ({s1}, {s2})"
                )

        n_samples = min(samples)
        n_channels = channels[0]
        stems_tracks = {s:t[:, :n_samples] for s,t in stems_tracks.items()}
        
        for stem in self.stems:
            if not stem in stems_tracks:
                stems_tracks[stem] = torch.zeros(n_channels, n_samples)
        
        return tuple([stems_tracks[stem] for stem in self.stems])

    @property
    def sample_rate(self) -> int:
        return self.sr

    def __getitem__(self, item):
        return self.get_tracks(self.tracks[item])


class ChunkedSupervisedDataset(SupervisedDataset):
    def __init__(
        self,
        audio_dir: Union[Path, str],
        stems: List[str],
        sample_rate: int,
        max_chunk_size: int,
        min_chunk_size: int,
    ):
        super().__init__(audio_dir=audio_dir, stems=stems, sample_rate=sample_rate)

        self.max_chunk_size = max_chunk_size
        self.available_chunk = {}
        self.index_to_track, self.index_to_chunk = [], []

        for track in self.tracks:
            tracks = self.get_tracks(track)
            available_chunks = get_nonsilent_chunks(sum(tracks), max_chunk_size, min_chunk_size)
            self.available_chunk[track] = available_chunks
            self.index_to_track.extend([track] * len(available_chunks))
            self.index_to_chunk.extend(available_chunks)

        assert len(self.index_to_chunk) == len(self.index_to_track)

    def __len__(self):
        return len(self.index_to_track)

    def get_chunk_track(self, item: int) -> str:
        return self.index_to_track[item]

    def get_chunk_indices(self, item: int) -> Tuple[int, int]:
        ci = self.index_to_chunk[item]
        return ci * self.max_chunk_size, (ci + 1) * self.max_chunk_size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        chunk_start, chunk_stop = self.get_chunk_indices(item)
        tracks = self.get_tracks(self.get_chunk_track(item))
        tracks = tuple([t[:, chunk_start:chunk_stop] for t in tracks])
        return tracks

@functools.lru_cache(128)
@torch.no_grad()
def load_audio_track(path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    return torchaudio.load(path)


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
