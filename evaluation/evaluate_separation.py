from collections import defaultdict
import json
from pathlib import Path
from pathlib import Path
import re
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
#import museval
import pandas as pd
import torch
import torchaudio
import torchmetrics.functional.audio as tma
#from evaluation.evaluate_separation import evaluate_data
from tqdm import tqdm
from torchaudio.transforms import Resample


def sdr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s_target = torch.norm(target, dim=-1)**2 + eps
    s_error = torch.norm(target - preds, dim=-1)**2 + eps
    return 10 * torch.log10(s_target/s_error)


def sisnr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha * target
    noise = target_scaled - preds
    s_target = torch.sum(target_scaled**2, dim=-1) + eps
    s_error = torch.sum(noise**2, dim=-1) + eps

    return 10 * torch.log10(s_target / s_error)


def get_rms(source_waveforms):
  """Return shape (source,) weights for signals that are nonzero."""
  return torch.sqrt(torch.mean(source_waveforms ** 2, dim=-1))
  #return source_norms <= 1e-8


def load_chunks(chunk_folder: Path) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], int]:
    original_tracks_and_rate = {ori.name.split(".")[0][3:]: torchaudio.load(ori) for ori in sorted(list(chunk_folder.glob("ori*.wav")))}
    separated_tracks_and_rate = {sep.name.split(".")[0][3:]: torchaudio.load(sep) for sep in sorted(list(chunk_folder.glob("sep*.wav")))}
    assert tuple(original_tracks_and_rate.keys()) == tuple(separated_tracks_and_rate.keys())

    original_tracks = {k:t for k, (t,_) in original_tracks_and_rate.items()}
    sample_rates_ori = [s for (_,s) in original_tracks_and_rate.values()]

    separated_tracks = {k:t for k, (t,_) in separated_tracks_and_rate.items()}
    sample_rates_sep = [s for (_,s) in separated_tracks_and_rate.values()]

    assert len({*sample_rates_ori, *sample_rates_sep}) == 1, print({*sample_rates_ori, *sample_rates_sep})
    assert len(original_tracks) == len(separated_tracks)
    sr = sample_rates_ori[0]

    return original_tracks, separated_tracks, sr


def evaluate_chunks(separation_path, filter_silence: bool = True, batch_size: int = 512, orig_sr: int = 44100, resample_sr: Optional[int] = None):
    print(separation_path)
    separation_folder = Path(separation_path)
    seps, oris, ms = defaultdict(list), defaultdict(list), []
    
    chunks = list(separation_folder.glob("*"))
    complete_results = defaultdict(list)

    resample_fn = Resample(orig_freq=orig_sr, new_freq=resample_sr) if resample_sr is not None else lambda x: x
    
    for ci, chunk_folder in enumerate(tqdm(chunks)):
        if not chunk_folder.is_dir():
            continue
        
        original_tracks, separated_tracks, sr = load_chunks(chunk_folder)
        assert sr == orig_sr, f"chunk [{chunk_folder.name}]: expected freq={orig_sr}, track freq={sr}"

        m = sum(original_tracks.values())

        for k in original_tracks.keys():
            ori_t = original_tracks[k]
            sep_t = separated_tracks[k]

            rms = get_rms(ori_t)
            if rms <= 1e-8 and filter_silence:
                ori_t[:] = torch.nan

            ori_t = resample_fn(ori_t)
            sep_t = resample_fn(sep_t)
            
            oris[k].append(ori_t)
            seps[k].append(sep_t)

        ms.append(resample_fn(m))
        
        if (ci+1) % batch_size == 0:
            oris = {k: torch.stack(t, dim=0) for k,t in oris.items()}
            seps = {k: torch.stack(t, dim=0) for k,t in seps.items()}
            ms = torch.stack(ms, dim=0)

            #results = {f"SISNRi_{k}": (si_snr_unreduced(seps[k], oris[k]) - si_snr_unreduced(ms, oris[k])).view(-1).tolist() for k in oris}
            results = {f"SDR_{k}": sdr(seps[k], oris[k]).view(-1).tolist() for k in oris}
            for k,v in results.items():
                complete_results[k].extend(v)
            
            seps, oris, ms = defaultdict(list), defaultdict(list), []

    df = pd.DataFrame(complete_results)
    return df#.to_dict()


def evaluate_tracks(separation_path, orig_sr: int = 44100, resample_sr: Optional[int] = None):
    separation_folder = Path(separation_path)
    assert separation_folder.exists(), separation_folder
    assert (separation_folder / "chunk_data.json").exists(), separation_folder

    with open(separation_folder / "chunk_data.json") as f:
        chunk_data = json.load(f)

    resample_fn = Resample(orig_freq=orig_sr, new_freq=resample_sr) if resample_sr is not None else lambda x: x

    track_to_chunks = defaultdict(list)
    for chunk_data in chunk_data:
        track = chunk_data["track"]
        chunk_idx = chunk_data["chunk_index"]
        start_sample = chunk_data["start_chunk_sample"]
        #start_sample_sec = chunk_data["start_chunk_seconds"]
        #assert abs(start_sample / orig_sr  - start_sample_sec) <= 1e-12, abs(start_sample / orig_sr  - start_sample_sec)
        track_to_chunks[track].append( (start_sample, chunk_idx) )

    # reorder chunks into ascending order and compute sdr
    track_to_sdr = {}
    for track, chunks in tqdm(track_to_chunks.items()):
        sorted_chunks = sorted(chunks)

        separated_wavs = defaultdict(list)
        original_wavs = defaultdict(list)

        for _, chunk_idx in sorted_chunks:
            chunk_folder = separation_folder / str(chunk_idx)
            original_tracks, separated_tracks, sr = load_chunks(chunk_folder)
            assert sr == orig_sr, f"chunk [{chunk_folder.name}]: expected freq={orig_sr}, track freq={sr}"

            for k in separated_tracks:
                separated_wavs[k].append(separated_tracks[k])
                original_wavs[k].append(original_tracks[k])

        for k in separated_wavs:
            separated_wavs[k] = resample_fn(torch.cat(separated_wavs[k], dim=-1))
            original_wavs[k] = resample_fn(torch.cat(original_wavs[k], dim=-1))

        mixture = sum([owav for owav in original_wavs.values()])

        #track_to_sdr[track] = {f"SDR_{k}": sdr(separated_wavs[k], original_wavs[k]).item() for k in separated_wavs}
        track_to_sdr[track] = {f"SISNRi_{k}": (sisnr(separated_wavs[k],  original_wavs[k]) - sisnr(mixture, original_wavs[k])).item() for k in separated_wavs}
    return pd.DataFrame.from_records(track_to_sdr).transpose()


def evaluate_tracks_chunks(separation_path, chunk_size: int, orig_sr: int = 44100, resample_sr: Optional[int] = None):

    separation_folder = Path(separation_path)
    assert separation_folder.exists(), separation_folder
    assert (separation_folder / "chunk_data.json").exists(), separation_folder

    with open(separation_folder / "chunk_data.json") as f:
        chunk_data = json.load(f)

    resample_fn = Resample(orig_freq=orig_sr, new_freq=resample_sr) if resample_sr is not None else lambda x: x

    track_to_chunks = defaultdict(list)
    for chunk_data in chunk_data:
        track = chunk_data["track"]
        chunk_idx = chunk_data["chunk_index"]
        start_sample = chunk_data["start_chunk_sample"]
        track_to_chunks[track].append( (start_sample, chunk_idx) )

    # reorder chunks into ascending order and compute sdr
    results = defaultdict(list)
    for track, chunks in tqdm(track_to_chunks.items()):
        sorted_chunks = sorted(chunks)

        separated_wavs, original_wavs = defaultdict(list), defaultdict(list)
        for _, chunk_idx in sorted_chunks:
            chunk_folder = separation_folder / str(chunk_idx)
            original_tracks, separated_tracks, sr = load_chunks(chunk_folder)
            assert sr == orig_sr, f"chunk [{chunk_folder.name}]: expected freq={orig_sr}, track freq={sr}"

            for k in separated_tracks:
                separated_wavs[k].append(separated_tracks[k])
                original_wavs[k].append(original_tracks[k])

        original_tensors, separated_tensors = {}, {}
        for k in separated_wavs:
            separated_tensors[k] = resample_fn(torch.cat(separated_wavs[k], dim=-1).view(-1))
            original_tensors[k] = resample_fn(torch.cat(original_wavs[k], dim=-1).view(-1))

        mixture = sum([owav for owav in original_tensors.values()])

        for i in range(mixture.shape[-1] // chunk_size):
            for k in separated_tensors:
                o = original_tensors[k][i*chunk_size:(i+1)*chunk_size]
                s = separated_tensors[k][i*chunk_size: (i+1)*chunk_size]
                m = mixture[i*chunk_size: (i+1)*chunk_size]
                results[k].append((sisnr(s, o) - sisnr(m, o)).item())

    return pd.DataFrame(results)

def read_ablation_results(sep_dir: Union[str, Path], filter_silence: bool = True):
    sep_dir = Path(sep_dir)

    hparams_files = list(sep_dir.glob("*.json"))
    records = []
    for hparam_path in tqdm(hparams_files):
        with open(hparam_path, "r") as f:
            hparams = json.load(f)
            assert isinstance(hparams, Mapping), type(hparams)

            match = re.fullmatch(
                "experiment-(?P<exp_num>[0-9]*)-hparams.json", hparam_path.name
            )
            assert match is not None
            experiment_number = match.groupdict()["exp_num"]
            experiment_dir = sep_dir / f"experiment-{experiment_number}"
            # print(experiment_dir)
            hparams_results = evaluate_chunks(experiment_dir, filter_silence=filter_silence)
            records.append({**hparams, **hparams_results})

    return pd.DataFrame.from_records(records)
