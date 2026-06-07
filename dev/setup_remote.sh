#!/bin/bash

# This script sets up the RL env needed for all experiments.


# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR


# -----------------------------------------------------------------------------
# Update for torch.compile
# Some older machines lack a Python.h file
apt-get update
apt-get install -y python3.10-dev build-essential
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Set api key: export WANDB_API_KEY=blah blah
# 2) Make sure to log in to wandb, e.g. run:
#    `wandb login`
# 3) Set the WANDB_RUN environment variable when running scripts, e.g.:
#    `WANDB_RUN=d32_grpo_orig python chat_rl.py`

# if [ -z "$WANDB_RUN" ]; then
#     # by default use "dummy" : it's handled as a special case, skips logging to wandb
#     WANDB_RUN=dummy
# fi

# -----------------------------------------------------------------------------
# report reset
# -----------------------------------------------------------------------------

# Download model and extra info from hf
mkdir -p $NANOCHAT_BASE_DIR/pretrained_checkpoints/
PT_DIR=$NANOCHAT_BASE_DIR/pretrained_checkpoints/nanochat-d32

# model details: https://github.com/karpathy/nanochat/discussions/8
hf download karpathy/nanochat-d32 --local-dir $PT_DIR

mkdir -p $NANOCHAT_BASE_DIR/tokenizer
mkdir -p $NANOCHAT_BASE_DIR/chatsft_checkpoints/d32

cp $PT_DIR/token_bytes.pt $PT_DIR/tokenizer.pkl \
  $NANOCHAT_BASE_DIR/tokenizer/

cp $PT_DIR/meta_000650.json $PT_DIR/model_000650.pt \
  $NANOCHAT_BASE_DIR/chatsft_checkpoints/d32/


# python -m scripts.chat_rl --run d32_gsm8k_orig --model-tag d32 --model-step 650 --device-batch-size 1