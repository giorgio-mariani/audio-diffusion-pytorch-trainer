import torchaudio
from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from audio_data_pytorch.utils import fractional_random_split
from torch.utils.data import DataLoader

from main.discriminator import ContextDiscriminator


""" Model """
class Model(pl.LightningModule):
    def __init__(
        self, learning_rate: float, beta1: float, beta2: float, *args, **kwargs
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = ContextDiscriminator(*args, **kwargs)  # AudioDiffusionModel(*args, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return optimizer

    def step(self, batch):
        waveforms = batch
        N, K, _ = waveforms.shape
        stft = torch.zeros(N, K, 257, 1025).type_as(waveforms)
        for n in range(N):
            for k in range(K):
                stft[n, k] = torch.stft(waveforms[n, k], n_fft=512, hop_length=256, win_length=256,
                                       window=torch.hann_window(window_length=256).type_as(stft),
                                       return_complex=True).abs()
        stft_real = stft[:N // 2]
        stft_fake = stft[N // 2:]
        perm = torch.randperm(N)[:K]
        idx = (torch.arange(N // 2).unsqueeze(-1).repeat((1, K)) + perm) % (N // 2)
        js = torch.arange(K).unsqueeze(0).repeat((N // 2,1))
        logits_real = self.model(stft_real)
        stft_fake = stft_fake[idx, js]
        logits_fake = self.model(stft_fake)
        loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
        loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
        loss = 0.5 * loss_real + 0.5 * loss_fake
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('valid_loss', loss)
        return loss


""" Datamodule """
class DatamoduleWithValidation(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.data_train = train_dataset
        self.data_val = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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


