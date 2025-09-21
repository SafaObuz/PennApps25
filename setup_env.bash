# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# CUDA Environment Setup
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=$CUDA_HOME
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

# You probably only need to change the following 1-2 lines
export EGOLIFTER_PATH=${HOME}/Code/egolifter/
export GSA_PATH=${EGOLIFTER_PATH}/Grounded-Segment-Anything/

# Paths related to GroundedSAM
export SAM_CHECKPOINT_PATH=/home/safaobuz/Code/Egocentric\ Perception\ Test/Grounded-Segment-Anything/sam_vit_b.pth
export GROUNDING_DINO_CHECKPOINT_PATH=/home/safaobuz/Code/Egocentric\ Perception\ Test/Grounded-Segment-Anything/groundingdino_swint_ogc.pth

# TODO: change below when we are actually going to use them
export TAG2TEXT_CHECKPOINT_PATH=${TAG2TEXT_PATH}/tag2text_swin_14m.pth
export RAM_CHECKPOINT_PATH=${TAG2TEXT_PATH}/ram_swin_large_14m.pth

# You probably won't need to change the following
export SAM_ENCODER_VERSION=vit_b
export GROUNDING_DINO_CONFIG_PATH=${GSA_PATH}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export EFFICIENTSAM_PATH=${GSA_PATH}/EfficientSAM/
export TAG2TEXT_PATH=${GSA_PATH}/Tag2Text/