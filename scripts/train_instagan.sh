#!/usr/bin/env bash

set -ex
no_proxy=localhost python train.py --dataroot ./datasets/pants2skirt_mhp --model insta_gan --name pants2skirt_mhp_instagan --loadSizeH 270 --loadSizeW 180 --fineSizeH 240 --fineSizeW 160 --display_id 1





# python -m visdom.server -p 8098

# ./scripts/train_instagan.sh



#python train.py --dataroot ./datasets/pants2skirt_mhp --model insta_gan --name pants2skirt_mhp_instagan --loadSizeH 270 --loadSizeW 180 --fineSizeH 240 --fineSizeW 160 --display_id 0 --tensorboardx --id [...]



# tensorboard --logdir checkpoints