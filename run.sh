#!/bin/bash
python train.py exp=base_slakh_1.yaml trainer.gpus=1 datamodule.batch_size=16 datamodule.dataset.path=/home/irene/Downloads/piano_22050/train
