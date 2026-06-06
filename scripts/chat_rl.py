# Make sure to setup WANDB
# export WANDB_API_KEY=blah blah
# wandb init
# python -m scripts.chat_rl

# assume 1 gpu

import pdb
import argparse
import numpy as np
import random
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8K

# ------------------------- Parser ---------------------------------------------------
parser = argparse.ArgumentParser(description="RL on GSM8k")
# Batch size
parser.add_argument("--device-batch-size", type=int, default=8,  help="The batch size seen by each GPU.")
parser.add_argument("--examples-per-step", type=int, default=16, help="The total number of examples seen in 1 step across all GPUs.")
parser.add_argument("--num-samples",       type=int, default=16, help="The total number of samples per prompt in a group.")
# Resume checkpoint details
parser.add_argument("--source",      type=str, default="sft", help="base/sft/rl. Stage of training.")
parser.add_argument("--step",        type=int, default=650,   help="Step to resume from.")
parser.add_argument("--device-type", type=str, default="cuda", help="Where to load the model.")

parser.add_argument("--dtype", type=str, default="bfloat16", help="The datatype to use.")

# sampling
parser.add_argument("--max_new_tokens", type=int, default="256", help="The number of new tokens during rollout.")
# ------------------------------------------------------------------------------------

args = parser.parse_args()
assert args.num_samples % args.device_batch_size == 0
dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

@torch.no_grad()
def get_rollout_group(model, task, engine, tokenizer, args, step):
    example_idxs_this_rank = list(range(len(task)))  # assume 1 gpu
    random.shuffle(example_idxs_this_rank)
    for idx in example_idxs_this_rank:
        model.eval()
        conversation = gsm8k[idx]
        tokens = tokenizer.render_for_completion(conversation)
        full_completions_group = []
        masks_group = []
        for sampling_microbatch in range(args.num_samples // args.device_batch_size):
            seed = hash((idx, step, sampling_microbatch)) & 0x7FFFFFFF  # hash can return negative int
            with autocast_ctx:
                full_completions_group_mb, masks_group_mb = engine.generate_batch(
                    tokens, 
                    num_samples=args.device_batch_size, 
                    max_tokens=args.max_new_tokens, 
                    temperature=1.0, 
                    top_k=None, 
                    seed=seed
                )
            full_completions_group.extend(full_completions_group_mb)
            masks_group.extend(masks_group_mb)
        
        generations_group = []
        rewards_group = []
        for full_completion in full_completions_group:
            generated_tokens = full_completion[len(tokens): ]
            response = tokenizer.decode(generated_tokens)
            reward = task.reward(conversation, response)
            generations_group.append(generated_tokens)
            rewards_group.append(reward)
        rewards = torch.tensor(rewards_group)
        mu = rewards.mean()
        advantages_group = rewards - mu

        # add assistant end token because engine.generate does not.
        # TODO: Investigate why does engine not generate the the end token? What's the advantage?
        # TODO: Why should we not take a loss over the end token?
        #   Is it to prevent the model from getting stuck at never predicting end = unending reasoning = 0 reward = 0 loss
        #   Seems like the idea is: teach the model reasoning strength during SFT and only modify the value of the tokens (not number) during RL.
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        max_len = max(len(full_completion) for full_completion in full_completions_group)
        full_completions_group = [
            full_completion + [assistant_end] * (max_len - len(full_completion))
            for full_completion in full_completions_group
        ]
        masks_group = [
            mask + [0] * (max_len - len(mask))
            for mask in masks_group
        ]
        completions_tensor = torch.tensor(full_completions_group)
        masks_tensor = torch.tensor(masks_group)
        assert completions_tensor.shape == masks_tensor.shape
        
        inputs = completions_tensor[:, : -1]
        targets = completions_tensor[:, 1: ]
        targets[masks_tensor[:, 1: ] == 0] = -1  # gpt.py sets ignore index to -1

        # send generations_group and rewards_group for logging only
        yield generations_group, inputs, targets, rewards_group, advantages_group

if __name__ == "__main__":
    gsm8k = GSM8K(subset="main", split="train")

    device = torch.device(args.device_type) # TODO: add rank info for ddp
    model, tokenizer, meta_data = load_model(source=args.source, phase="eval", step=args.step, device=device)
    engine = Engine(model, tokenizer)
    step = 0
    for rollout_group in get_rollout_group(model, gsm8k, engine, tokenizer, args, step):
        generations_group, inputs, targets, rewards_group, advantages_group = rollout_group
        pdb.set_trace()

    # setup optimizer
    # loop over batches
    # compute loss for each batch
    # backward
    # opt step
    # logging info