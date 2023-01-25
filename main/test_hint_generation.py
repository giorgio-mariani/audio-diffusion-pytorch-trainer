import torchaudio
import math
import sys
sys.path.append("/home/irene/Documents/audio-diffusion-pytorch-trainer")

from main.separation import ContextualSeparator
from pathlib import Path
from audio_diffusion_pytorch import KarrasSchedule
import torch
from script.misc import hparams
import main.module_base
device = torch.device("cuda")
ROOT_PATH = Path(".")

sampling_rate = 22050

# @markdown Generation length in seconds (will be rounded to be a power of 2 of sample_rate*length)
length = 10
length_samples = 2**math.ceil(math.log2(length * sampling_rate))

# @markdown Number of samples to generate
num_samples = 1

# @markdown Number of diffusion steps (higher tends to be better but takes longer to generate)
num_steps = 100
num_stems = 4

s1, sr1 = torchaudio.load('/home/irene/Documents/audio-diffusion-pytorch-trainer/data/Slack/test/bass/Track01881.wav')
s2, sr2 = torchaudio.load('/home/irene/Documents/audio-diffusion-pytorch-trainer/data/Slack/test/drums/Track01881.wav')
s3, sr3 = torchaudio.load('/home/irene/Documents/audio-diffusion-pytorch-trainer/data/Slack/test/guitar/Track01881.wav')
s4, sr4 = torchaudio.load('/home/irene/Documents/audio-diffusion-pytorch-trainer/data/Slack/test/piano/Track01881.wav')

s1 = torchaudio.functional.resample(s1, orig_freq=sr1, new_freq=sampling_rate)
s2 = torchaudio.functional.resample(s2, orig_freq=sr1, new_freq=sampling_rate)
s4 = torchaudio.functional.resample(s3, orig_freq=sr1, new_freq=sampling_rate)
s5 = torchaudio.functional.resample(s4, orig_freq=sr1, new_freq=sampling_rate)

# display(Audio(s1, rate = sampling_rate))
# display(Audio(s2, rate = sampling_rate))

start_sample = 100 * sampling_rate
s1 = s1.reshape(1, 1, -1)[:, :, start_sample:start_sample + length_samples]
s2 = s2.reshape(1, 1, -1)[:, :, start_sample:start_sample + length_samples]
s3 = s3.reshape(1, 1, -1)[:, :, start_sample:start_sample + length_samples]
s4 = s4.reshape(1, 1, -1)[:, :, start_sample:start_sample + length_samples]
m = s1+s2+s3+s4

smin = 1e-4
smax = 1.0
rho = 7.0
sigma_schedule=KarrasSchedule(sigma_min=smin, sigma_max=smax, rho=rho)

inpaint                = torch.randn(num_samples, num_stems, length_samples).to("cuda")
inpaint[:, 0, :]       = s1
inpaint_mask           = torch.ones_like(inpaint)
inpaint_mask[:, 1:, :] = 0.

def load_model(path):
  model = main.module_base.Model(**{**hparams, "in_channels": 4})
  model.load_state_dict(torch.load(path))
  model.to(device);
  return model

model = load_model(ROOT_PATH / "logs/ckpts/avid-darkness-164_epoch=311.ckpt")
separator = ContextualSeparator(model=model, stems=["bass", "drums", "guitar", "piano"], sigma_schedule=sigma_schedule)
separator.separate_with_hint(
    mixture=m,
    source_with_hint=inpaint,
    mask=inpaint_mask,
    num_steps=num_steps
)
print("fatto")