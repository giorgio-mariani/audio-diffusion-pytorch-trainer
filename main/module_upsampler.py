import random
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import librosa
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from audio_data_pytorch.utils import fractional_random_split
from audio_diffusion_pytorch import AudioDiffusionUpsampler, Sampler, Schedule
from audio_diffusion_pytorch.utils import downsample, upsample
from einops import rearrange
from ema_pytorch import EMA
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_eps: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        upsampler: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_eps = lr_eps
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_weight_decay = lr_weight_decay
        self.model = upsampler
        self.model_ema = EMA(self.model, beta=ema_beta, power=ema_power)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        waveforms = batch
        loss = self.model(waveforms)
        self.log("train_loss", loss)
        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay())
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms = batch
        loss = self.model_ema(waveforms)
        self.log("valid_loss", loss)
        return loss


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        val_split: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        split = [1.0 - self.val_split, self.val_split]
        self.data_train, self.data_val = fractional_random_split(self.dataset, split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    print("WandbLogger not found.")
    return None


def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )


def log_wandb_audio_spectrogram(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = samples.detach().cpu()
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        center=True,
        norm="slaney",
    )

    def get_spectrogram_image(x):
        spectrogram = transform(x[0])
        image = librosa.power_to_db(spectrogram)
        trace = [go.Heatmap(z=image, colorscale="viridis")]
        layout = go.Layout(
            yaxis=dict(title="Mel Bin (Log Frequency)"),
            xaxis=dict(title="Frame"),
            title_text=caption,
            title_font_size=10,
        )
        fig = go.Figure(data=trace, layout=layout)
        return fig

    logger.log(
        {
            f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
            for idx in range(num_items)
        }
    )


def to_list(val):
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        factor: Union[int, Sequence[int]],
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
    ) -> None:
        self.num_items = num_items
        self.factors = to_list(factor)
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.sampling_steps = sampling_steps
        self.diffusion_schedule = diffusion_schedule
        self.diffusion_sampler = diffusion_sampler

        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        model = pl_module.model_ema.ema_model

        # Log true waveforms
        waveforms = batch[0 : self.num_items]
        log_wandb_audio_batch(
            logger=wandb_logger,
            id="true",
            samples=waveforms,
            sampling_rate=self.sampling_rate,
        )
        log_wandb_audio_spectrogram(
            logger=wandb_logger,
            id="true",
            samples=waveforms,
            sampling_rate=self.sampling_rate,
        )

        # Compute and log downsampled waveforms
        factor = random.choice(self.factors)
        downsampled_rate = self.sampling_rate // factor
        waveforms_downsampled = downsample(waveforms, factor=factor)
        # We log an upsampled version since the player doesn't support low Hz rates
        waveforms_reupsampled = upsample(waveforms_downsampled, factor=factor)
        log_wandb_audio_batch(
            logger=wandb_logger,
            id="downsampled",
            samples=waveforms_reupsampled,
            sampling_rate=self.sampling_rate,
            caption=f"Sample rate {downsampled_rate}",
        )

        # Log upsampled waveforms at different steps
        for steps in self.sampling_steps:
            samples = model.sample(
                waveforms_downsampled,
                factor=factor,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
            )
            log_wandb_audio_batch(
                logger=wandb_logger,
                id="upsampled",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps from {downsampled_rate} Hz",
            )
            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id="upsampled",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps from {downsampled_rate} Hz",
            )

        if is_train:
            pl_module.train()
