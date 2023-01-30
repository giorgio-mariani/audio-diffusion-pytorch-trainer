import sys
from pathlib import Path
import torch
import torchaudio as ta
import torchmetrics
from audio_diffusion_pytorch import LogNormalDistribution
from IPython.display import Audio


ROOT_PATH=Path("..").resolve().absolute()
sys.path.append(str(ROOT_PATH))

import main.module_base

diffusion_sigma_distribution = LogNormalDistribution(mean=-3.0, std=1.0)
hparams = dict(    
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.99,
    in_channels=1,
    channels=256,
    patch_factor=16,
    patch_blocks=1,
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    kernel_sizes_init=[1, 3, 7],
    multipliers=[1, 2, 4, 4, 4, 4, 4],
    factors=[4, 4, 4, 2, 2, 2],
    num_blocks= [2, 2, 2, 2, 2, 2],
    attentions= [False, False, False, True, True, True],
    attention_heads=8,
    attention_features=128,
    attention_multiplier=2,
    use_nearest_upsample=False,
    use_skip_scale=True,
    use_attention_bottleneck=True,
    diffusion_sigma_distribution=diffusion_sigma_distribution,
    diffusion_sigma_data=0.2,
    diffusion_dynamic_threshold=0.0,
)

hparams_3stems = dict(
  learning_rate= 1e-4,
  beta1= 0.9,
  beta2= 0.99,
  in_channels= 3,
  channels= 288,
  patch_factor= 16,
  patch_blocks= 1,
  resnet_groups= 8,
  kernel_multiplier_downsample= 2,
  kernel_sizes_init= [1, 3, 7],
  multipliers= [1, 2, 4, 4, 4, 4, 4],
  factors= [4, 4, 4, 2, 2, 2],
  num_blocks= [2, 2, 2, 2, 2, 2],
  attentions=[False, False, False, True, True, True],
  attention_heads=8,
  attention_features=128,
  attention_multiplier=2,
  use_nearest_upsample=False,
  use_skip_scale=True,
  use_attention_bottleneck=True,
  diffusion_sigma_distribution=diffusion_sigma_distribution,
  diffusion_sigma_data=0.2,
  diffusion_dynamic_threshold=0.0,
)

def load_model(path: str, device: str) -> main.module_base.Model:
    model_1 = main.module_base.Model(**hparams)
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    model_1.load_state_dict(ckpt["state_dict"])
    model_1.to(device)
    return model_1

def load_context(path: str, device: str, num_sources: int) -> main.module_base.Model:
    model = main.module_base.Model(**{**hparams, "in_channels": num_sources})
    model.load_state_dict(torch.load(path)["state_dict"])
    model.to(device)
    return model


def load_audio(path: str, sample_rate: int, start: int = 0, end: int = -1):
    s1, sr1 = ta.load(path)
    s1 = ta.functional.resample(s1, orig_freq=sr1, new_freq=sample_rate)
    return s1.unsqueeze(0)[:, :, start:end]


def display_audio(signal, sr):
    display(Audio(signal.squeeze(0).cpu(), rate=sr))
    
    
# metrics ================

def si_snr(preds, targets):
    return torchmetrics.functional.audio.scale_invariant_signal_distortion_ratio(preds.cpu(), targets.cpu()).item()


def sdr(preds, targets):
    preds = preds.cpu()
    targets = targets.cpu()

    signal = torch.norm(targets) ** 2
    e_error = torch.norm(preds - targets) ** 2
    return 10 * torch.log10( signal/e_error )