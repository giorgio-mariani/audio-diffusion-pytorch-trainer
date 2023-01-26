from collections import defaultdict
import json
from pathlib import Path
from pathlib import Path
import re
from typing import Mapping, Optional, Union

import numpy as np
#import museval
import pandas as pd
import torch
import torchaudio
import torchmetrics.functional.audio as tma
#from evaluation.evaluate_separation import evaluate_data
from tqdm import tqdm
from torchaudio.transforms import Resample


def si_snr(preds: torch.Tensor, target: torch.Tensor) -> float:
    return tma.scale_invariant_signal_noise_ratio(preds=preds.cpu(), target=target.cpu()).mean().item()

def si_snr_unreduced(preds: torch.Tensor, target: torch.Tensor) -> float:
    return tma.scale_invariant_signal_noise_ratio(preds=preds.cpu(), target=target.cpu())


def si_sdr(preds: torch.Tensor, target: torch.Tensor) -> float:
    return tma.scale_invariant_signal_distortion_ratio(preds=preds.cpu(), target=target.cpu()).mean().item()


def sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10((torch.norm(target, dim=-1)**2 + 1e-7)/((torch.norm(target - preds, dim=-1))**2 + 1e-7))


def museval_sdr(preds: torch.Tensor, target: torch.Tensor, sample_rate: int) -> float:
    """"""
    preds = preds.cpu()
    target = target.cpu()
    assert target.shape == preds.shape
    batch_size, num_src, num_channels, num_samples = preds.shape
    #target, preds = target.permute(dims=[0, 1, 3, 2]), preds.permute(dims=[0, 1, 3, 2])

    batch_sdr = []
    for i in range(batch_size):
        t, p = target[i].permute([0, 2, 1]), preds[i].permute([0, 2, 1])
        (
            sdr_metric,
            isr_metric,
            sir_metric,
            sar_metric,
        ) = museval.evaluate(references=t, estimates=p, win=sample_rate, hop=sample_rate)
        sdr_metric[sdr_metric == np.inf] = np.nan
        batch_sdr.append(np.nanmedian(sdr_metric))
    return sum(batch_sdr) / batch_size


def get_rms(source_waveforms):
  """Return shape (source,) weights for signals that are nonzero."""
  return torch.sqrt(torch.mean(source_waveforms ** 2, dim=-1))
  #return source_norms <= 1e-8


def evaluate_data(separation_path, filter_silence: bool = True, batch_size: int = 512, orig_sr: int = 44100, resample_sr: Optional[int] = None):
    separation_folder = Path(separation_path)
    seps, oris, ms = defaultdict(list), defaultdict(list), []
    
    chunks = list(separation_folder.glob("*"))
    complete_results = defaultdict(list)

    resample_fn = Resample(orig_freq=orig_sr, new_freq=resample_sr) if resample_sr is not None else lambda x: x
    
    for ci, chunk_folder in enumerate(tqdm(chunks)):
        if not chunk_folder.is_dir():
            continue
        
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

            results = {f"SISNRi_{k}": (si_snr_unreduced(seps[k], oris[k]) - si_snr_unreduced(ms, oris[k])).view(-1).tolist() for k in oris}
            for k,v in results.items():
                complete_results[k].extend(v)
            
            seps, oris, ms = defaultdict(list), defaultdict(list), []

    df = pd.DataFrame(complete_results).mean()
    return df#.to_dict()

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
            hparams_results = evaluate_data(experiment_dir, filter_silence=filter_silence)
            records.append({**hparams, **hparams_results})

    return pd.DataFrame.from_records(records)
