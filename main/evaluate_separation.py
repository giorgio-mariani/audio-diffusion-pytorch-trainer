from pathlib import Path

import museval
import tqdm

import torch
import torchaudio
import numpy as np

import torchmetrics.functional.audio as tma


def si_snr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return tma.scale_invariant_signal_noise_ratio(preds=preds.cpu(), target=target.cpu()).mean().item()


def si_sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return tma.scale_invariant_signal_distortion_ratio(preds=preds.cpu(), target=target.cpu()).mean().item()


def sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    seps1, seps2, oris1, oris2, ms = [], [], [], [], []

    for chunk_folder in (list(separation_folder.glob("*"))):
        original_tracks = [torchaudio.load(ori) for ori in sorted(list(chunk_folder.glob("ori*.wav")))]
        separated_tracks = [torchaudio.load(sep) for sep in sorted(list(chunk_folder.glob("sep*.wav")))]

        original_tracks, sample_rates_ori = zip(*original_tracks)
        separated_tracks, sample_rates_sep = zip(*separated_tracks)

        assert len({*sample_rates_ori, *sample_rates_sep}) == 1
        sample_rate = sample_rates_ori[0]

        assert len(original_tracks) == len(separated_tracks)
        assert len(original_tracks) == 2
        ori1, ori2 = original_tracks
        sep1, sep2 = separated_tracks

        m = ori1 + ori2

        if torch.amax(torch.abs(ori1)) < 1e-3 or torch.amax(torch.abs(ori2)) < 1e-3:
            continue

        seps1.append(sep1)
        seps2.append(sep2)
        oris1.append(ori1)
        oris2.append(ori2)
        ms.append(m)

    seps1 = torch.stack(seps1, dim=0)
    seps2 = torch.stack(seps2, dim=0)
    seps = torch.stack([seps1, seps2], dim=1)

    oris1 = torch.stack(oris1, dim=0)
    oris2 = torch.stack(oris2, dim=0)
    oris = torch.stack([oris1, oris2], dim=1)

    ms = torch.stack(ms, dim=0)
    results = {
        "sisnr_1": si_snr(seps1, oris1),
        "sisnr_2": si_snr(seps2, oris2),
        "sisnri_1": si_snr(seps1, oris1) - si_snr(ms, oris1),
        "sisnri_2": si_snr(seps2, oris2) - si_snr(ms, oris2),
        "sdr": sdr(seps, oris),
    }
    return results

    #print("SI-SNR 1: ", si_snr(seps1, oris1))
    #print("SI-SNR 2: ", si_snr(seps2, oris2))
    #print("SI-SNRi 1: ", si_snr(seps1, oris1) - si_snr(ms, oris1))
    #print("SI-SNRi 2: ", si_snr(seps2, oris2) - si_snr(ms, oris2))
    #print("SDR mean: ", sdr(seps, oris))

#if __name__ == "__main__":
#    evaluate_separation()