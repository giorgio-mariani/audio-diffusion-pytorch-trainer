from collections import defaultdict
import itertools
import json
import os
from pathlib import Path
from pathlib import Path
import re
from typing import Callable, List, Mapping, Optional, Tuple, Union
import main.module_base
from script.misc import hparams
import math

#import museval
import pandas as pd
import torch
import torchaudio
#from evaluation.evaluate_separation import evaluate_data
from tqdm import tqdm
from torchaudio.transforms import Resample

from main.dataset import is_silent
from main.likelihood import log_likelihood_song


def sdr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s_target = torch.norm(target, dim=-1)**2 + eps
    s_error = torch.norm(target - preds, dim=-1)**2 + eps
    return 10 * torch.log10(s_target/s_error)


def sisnr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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


def evaluate_chunks(separation_path: Union[str, Path], filter_silence: bool = True, batch_size: int = 512, orig_sr: int = 44100, resample_sr: Optional[int] = None):
    separation_folder = Path(separation_path)
    seps, oris, ms = defaultdict(list), defaultdict(list), []
    
    chunks = list(separation_folder.glob("*"))
    complete_results = defaultdict(list)

    resample_fn = Resample(orig_freq=orig_sr, new_freq=resample_sr) if resample_sr is not None else lambda x: x
    for ci, chunk_folder in enumerate((chunks)):
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
        
        if (ci+1) % batch_size == 0 or (ci+1) == len(chunks):
            oris = {k: torch.stack(t, dim=0) for k,t in oris.items()}
            seps = {k: torch.stack(t, dim=0) for k,t in seps.items()}
            ms = torch.stack(ms, dim=0)

            results = {f"SISNRi_{k}": (sisnr(seps[k], oris[k]) - sisnr(ms, oris[k])).view(-1).tolist() for k in oris}
            #results = {f"SDR_{k}": sdr(seps[k], oris[k]).view(-1).tolist() for k in oris}
            for k,v in results.items():
                complete_results[k].extend(v)
            
            seps, oris, ms = defaultdict(list), defaultdict(list), []

    df = pd.DataFrame(complete_results).mean()
    return df.to_dict()


def evaluate_tracks(
    separation_path: Union[str, Path], 
    orig_sr: int = 44100, 
    resample_sr: Optional[int] = None,
    metrics_fns: Mapping[str, Callable] = None,
    ):
    
    metrics_fns = {"SDR": sdr} if metrics_fns is None else metrics_fns
    
    separation_folder = Path(separation_path)
    assert separation_folder.exists(), separation_folder
    assert (separation_folder.parent / "chunk_data.json").exists(), separation_folder

    with open(separation_folder.parent / "chunk_data.json") as f:
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
    track_to_metrics = {}
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

        sep_per_metric = itertools.product(separated_wavs, metrics_fns.items())
        track_to_metrics[track] = {f"{n}_{k}": fn(separated_wavs[k], original_wavs[k]) for k, (n, fn) in sep_per_metric}
    return pd.DataFrame.from_records(track_to_metrics).transpose()


def evaluate_tracks_chunks(separation_path: Union[str, Path], chunk_prop: int, 
                           orig_sr: int = 44100, resample_sr: Optional[int] = None, 
                           filter_single_source: bool = True, eps: float = 10-8):

    separation_folder = Path(separation_path)
    assert separation_folder.exists(), separation_folder
    assert (separation_folder.parent / "chunk_data.json").exists(), separation_folder

    with open(separation_folder.parent / "chunk_data.json") as f:
        chunk_data = json.load(f)
        
    def load_model(path):
        model = main.module_base.Model(**{**hparams, "in_channels": 4})
        model.load_state_dict(torch.load(path)["state_dict"])
        model.to("cuda:0")
        return model
    
    ckpts_path = Path("/home/irene/Documents/audio-diffusion-pytorch-trainer/logs/ckpts")
    model = load_model(ckpts_path / "avid-darkness-164_epoch=419-valid_loss=0.015.ckpt")
    denoise_fn = model.model.diffusion.denoise_fn

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
        chunk_size = int(separated_tensors["1"].shape[0] * chunk_prop)

        generated_mixture = torch.stack([separated_tensors[k] for k in separated_tensors]).unsqueeze(0).to("cuda:0")
        for k in separated_tensors:
            o = original_tensors[k]
            s = separated_tensors[k]
            m = mixture
            padded_source = torch.zeros((1, 4, s.shape[-1]))
            j = int(k) -1
            padded_source[:, j:j+1, :] = s
            lik = torch.zeros(1)
            #lik, _, _ = log_likelihood_song(denoise_fn, padded_source.to("cuda:0"), sigma_max=1.0) # Hard coded sigma_max
            #for i in range(mixture.shape[-1] // chunk_size):
            #    results[f"log_lik_{k}"].append(lik.item())
        #lik, _, _ = log_likelihood_song(denoise_fn, generated_mixture, sigma_max=1.0)
        
        #for i in range(mixture.shape[-1] // chunk_size):
        #    results[f"log_lik_mixture"].append(lik.item())
            
        for i in range(mixture.shape[-1] // chunk_size):
            num_silent_signals = 0
            for k in separated_tensors:
                o = original_tensors[k][i*chunk_size:(i+1)*chunk_size]
                if is_silent(o.unsqueeze(0)) and filter_single_source:
                    num_silent_signals += 1
            if num_silent_signals > 3:
                continue
            else:
                for k in separated_tensors:
                    o = original_tensors[k][i*chunk_size:(i+1)*chunk_size]
                    s = separated_tensors[k][i*chunk_size: (i+1)*chunk_size]
                    m = mixture[i*chunk_size: (i+1)*chunk_size]
                    results[k].append((sisnr(s, o, eps) - sisnr(m, o, eps)).item())
    #print(results)
    return pd.DataFrame(results)

def evaluate_tracks_chunks_simplified(separation_path: Union[str, Path],
                           orig_sr: int = 44100, resample_sr: Optional[int] = None, 
                           filter_single_source: bool = True, eps: float = 10-8, compute_likelihood = False,
                           chunk_duration: float = 4.0, overlap_duration: float = 2.0):

    separation_folder = Path(separation_path)
    assert separation_folder.exists(), separation_folder
    #assert (separation_folder / "chunk_data.json").exists(), separation_folder
    assert (separation_folder.parent / "chunk_data.json").exists(), separation_folder

    with open(separation_folder.parent / "chunk_data.json") as f:
        chunk_data = json.load(f)
        
    def load_model(path):
        model = main.module_base.Model(**{**hparams, "in_channels": 4})
        model.load_state_dict(torch.load(path)["state_dict"])
        model.to("cuda:0")
        return model
    
    ckpts_path = Path("/home/irene/Documents/audio-diffusion-pytorch-trainer/logs/ckpts")
    model = load_model(ckpts_path / "avid-darkness-164_epoch=419-valid_loss=0.015.ckpt")
    denoise_fn = model.model.diffusion.denoise_fn
    
    track_to_chunks = defaultdict(list)
    for chunk_data in chunk_data:
        track = chunk_data["track"]
        chunk_idx = chunk_data["chunk_index"]
        start_sample = chunk_data["start_chunk_sample"]
        track_to_chunks[track].append( (start_sample, chunk_idx) )

    # reorder chunks into ascending order and compute sdr
    results = defaultdict(list)
    # for every iteration, chunks contains the list of chunks associated to a track 
    for chunk_folder in tqdm(list(separation_path.iterdir())):
                        
        try:
            original_tracks, separated_tracks, sr = load_chunks(chunk_folder)   
            assert sr == orig_sr, f"chunk [{chunk_folder.name}]: expected freq={orig_sr}, track freq={sr}"   
            mixture = sum([owav for owav in original_tracks.values()])
            
            chunk_samples = chunk_duration * orig_sr
            overlap_samples = overlap_duration * orig_sr
            
            chunk_samples = int(chunk_duration * orig_sr)
            
            overlap_samples = int(overlap_duration * orig_sr)

            # Calculate the step size between consecutive chunks
            step_size = chunk_samples - overlap_samples

            # Determine the number of chunks based on step_size
            num_chunks = math.ceil((mixture.shape[-1] - overlap_samples) / step_size)
            #print(mixture.shape)
            
            if compute_likelihood:
                compute_likelihood_value(separated_tracks, denoise_fn, num_chunks)
                
            for i in range(num_chunks):
                start_sample = i * step_size
                end_sample = start_sample + chunk_samples
                
                num_silent_signals = 0
                
                for k in separated_tracks:
                    o = original_tracks[k][:,start_sample:end_sample]
                
                    if is_silent(o) and filter_single_source:
                        num_silent_signals += 1
                
                if num_silent_signals > 2:
                    continue
                else:
                    for k in separated_tracks:
                        o = original_tracks[k][:,start_sample:end_sample]
                        s = separated_tracks[k][:,start_sample:end_sample]
                        m = mixture[:,start_sample:end_sample]
                        results[k].append((sisnr(s, o, eps) - sisnr(m, o, eps)).item())
                    results["chunk_n"].append(chunk_folder.name)
                    results["start_sample"].append(start_sample)
                    results["end_sample"].append(end_sample)
        except:
            print("Orrore!")
            continue
    return pd.DataFrame(results)


def compute_likelihood_value(separated_tracks, denoise_fn, num_chunks, results):
    separated_tensor = torch.stack([separated_tracks[k] for k in separated_tracks]).to("cuda:0")
    separated_tensor = separated_tensor.permute(1, 0, 2)
    for k in separated_tracks:
        s = separated_tracks[k]
        padded_source = torch.zeros((1, 4, s.shape[-1]))
        j = int(k) -1
        padded_source[:, j:j+1, :] = s
        lik, _, _ = log_likelihood_song(denoise_fn, padded_source.to("cuda:0"), sigma_max=1.0) # Hard coded sigma_max
        for _ in range(num_chunks):
            results[f"log_lik_{k}"].append(lik.item())
    lik, _, _ = log_likelihood_song(denoise_fn, separated_tensor, sigma_max=1.0)
    
    for _ in range(num_chunks):
        results[f"log_lik_mixture"].append(lik.item())
    
def read_ablation_results(sep_dir: Union[str, Path], orig_sr: int = 44100, filter_silence: bool = False):
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
            chunk_results = evaluate_chunks(experiment_dir, orig_sr=orig_sr, filter_silence=filter_silence)
            hparams_results = pd.DataFrame(chunk_results).mean().to_dict()

            records.append({**hparams, **hparams_results})

    return pd.DataFrame.from_records(records)

if __name__ == "__main__":  
    print(f"{os.getcwd()=}")
    results = evaluate_tracks_chunks_simplified(Path("separations/complete_context_slakh_4stems_resamples"), 
                                                orig_sr=22050, eps=1e-8,
                                                compute_likelihood=False)
    print("bau")
    '''
    results = evaluate_tracks(
        Path("separations/context_musdb_full_steps=150_res=2_source_id=-1/sep_round_0"),
        orig_sr=16000,
    )
    print(results.mean())'''
