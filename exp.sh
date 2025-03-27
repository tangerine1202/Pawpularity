#!/bin/bash

set -x;

# model='ResNet18'
model='Swin_V2_B'
epochs=10
lr=3e-4
batch_size=16


python3 main.py TA -aug TA -m $model -lr $lr -ep $epochs -bs $batch_size
python3 main.py RA -aug RA -m $model -lr $lr -ep $epochs -bs $batch_size
python3 main.py AA_img -aug AA_img -m $model -lr $lr -ep $epochs -bs $batch_size
python3 main.py AA_svhn -aug AA_svhn -m $model -lr $lr -ep $epochs -bs $batch_size
python3 main.py AA_cifar10 -aug AA_cifar10 -m $model -lr $lr -ep $epochs -bs $batch_size