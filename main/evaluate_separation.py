from collections import defaultdict
from pathlib import Path

import museval
import tqdm

import torch
import torchaudio
import numpy as np

import torchmetrics.functional.audio as tma


def si_snr(preds: torch.Tensor, target: torch.Tensor) -> float:
    return tma.scale_invariant_signal_noise_ratio(preds=preds.cpu(), target=target.cpu()).mean().item()


def si_sdr(preds: torch.Tensor, target: torch.Tensor) -> float:
    return tma.scale_invariant_signal_distortion_ratio(preds=preds.cpu(), target=target.cpu()).mean().item()


def sdr(preds: torch.Tensor, target: torch.Tensor) -> float:
     return tma.signal_distortion_ratio(preds=preds.cpu(), target=target.cpu()).mean().item()


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


def evaluate_data(separation_path):
    separation_folder = Path(separation_path)
    seps, oris, ms = defaultdict(list), defaultdict(list), []

    for chunk_folder in (list(separation_folder.glob("*"))):
        original_tracks_and_rate = {ori.name.split(".")[0][3:]: torchaudio.load(ori) for ori in sorted(list(chunk_folder.glob("ori*.wav")))}
        separated_tracks_and_rate = {sep.name.split(".")[0][3:]: torchaudio.load(sep) for sep in sorted(list(chunk_folder.glob("sep*.wav")))}
        assert tuple(original_tracks_and_rate.keys()) == tuple(separated_tracks_and_rate.keys())

        original_tracks = {k:t for k, (t,_) in original_tracks_and_rate.items()}
        sample_rates_ori = [s for (_,s) in original_tracks_and_rate.values()]

        separated_tracks = {k:t for k, (t,_) in separated_tracks_and_rate.items()}
        sample_rates_sep = [s for (_,s) in separated_tracks_and_rate.values()]

        assert len({*sample_rates_ori, *sample_rates_sep}) == 1
        assert len(original_tracks) == len(separated_tracks)
        m = sum(original_tracks.values())

        #TODO: check silence
        #if torch.amax(torch.abs(ori1)) < 1e-3 or torch.amax(torch.abs(ori2)) < 1e-3:
        #    continue

        for k,t in original_tracks.items():
            oris[k].append(t)

        for k,t in separated_tracks.items():
            seps[k].append(t)

        ms.append(m)

    oris = {k: torch.stack(t, dim=0) for k,t in oris.items()}
    seps = {k: torch.stack(t, dim=0) for k,t in seps.items()}
    ms = torch.stack(ms, dim=0)

    # results = {
    #     "sisnr_1": si_snr(seps1, oris1),
    #     "sisnr_2": si_snr(seps2, oris2),
    #     "sisnri_1": si_snr(seps1, oris1) - si_snr(ms, oris1),
    #     "sisnri_2": si_snr(seps2, oris2) - si_snr(ms, oris2),
    #     "sdr": sdr(seps, oris),
    # }
    results = {f"SISNRi_{k}": si_snr(seps[k], oris[k]) - si_snr(ms, oris[k]) for k in oris}
    return results

    #print("SI-SNR 1: ", si_snr(seps1, oris1))
    #print("SI-SNR 2: ", si_snr(seps2, oris2))
    #print("SI-SNRi 1: ", si_snr(seps1, oris1) - si_snr(ms, oris1))
    #print("SI-SNRi 2: ", si_snr(seps2, oris2) - si_snr(ms, oris2))
    #print("SDR mean: ", sdr(seps, oris))

#if __name__ == "__main__":
#    evaluate_separation()
