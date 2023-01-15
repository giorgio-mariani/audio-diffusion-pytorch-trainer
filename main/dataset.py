import functools
import math
from pathlib import Path
from typing import Union, Optional, Callable, Tuple

import random
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
        num_channels, _ = wav.shape
        if self.training:
            signs = torch.randint(2, (num_channels,1), device=wav.device, dtype=torch.float32)
            wav = wav * (2 * signs - 1)
        return wav


class Scale(nn.Module):
    def __init__(self, scales_sampler: Distribution, proba=1.):
        super().__init__()
        self.proba = proba
        self.scales_sampler = scales_sampler

    def forward(self, wav):
        num_channels, _ = wav.shape
        if self.training and random.random() < self.proba:
            wav[:, :] = wav[:, :] * self.scales_sampler.sample([num_channels, 1])
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
