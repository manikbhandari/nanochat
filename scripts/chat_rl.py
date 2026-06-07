# Make sure to setup WANDB
# export WANDB_API_KEY=blah blah
# wandb init
# python -m scripts.chat_rl
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl

# assume 1 gpu

import argparse
import functools
import itertools
import os
import pdb
import random
import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# ------------------------- Parser ---------------------------------------------------
parser = argparse.ArgumentParser(description="RL on GSM8k")
parser.add_argument("--debug", action="store_true", help="Add debugging logging.")
parser.add_argument("--no-wandb", action="store_true", help="Don't log to wandb.")
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

parser.add_argument("--epochs", type=int, default=1, help="Total epochs over the task.")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="Initial learning rate gets multiplied by this.")
# ------------------------------------------------------------------------------------

args = parser.parse_args()
assert args.num_samples % args.device_batch_size == 0  # TODO: relax this assumption for num_samples < batch_size
dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
wandb_run = None if (args.no_wandb or int(os.environ.get("RANK", 0)) != 0) else wandb.init(project="nanochat-rl", name="d32_gsm8k_redo_8gpu", config=vars(args).copy())

def print0(s):
    if int(os.environ.get("RANK", 0)) == 0:
        print(s)

def printd(s):
    if args.debug:
        print0(s)

@torch.no_grad()
def get_rollout_group(model, task, engine, tokenizer, args, ddp_world_size, step):
    example_idxs_this_rank = list(range(0, len(task), ddp_world_size))  # assume 1 gpu
    random.shuffle(example_idxs_this_rank)
    for idx in itertools.cycle(example_idxs_this_rank):
        model.eval()
        conversation = gsm8k[idx]
        tokens = tokenizer.render_for_completion(conversation)
        full_completions_group = []
        masks_group = []
        n_microbatches = max(args.num_samples // args.device_batch_size, 1)
        for sampling_microbatch in range(n_microbatches):
            seed = hash((idx, step, sampling_microbatch)) & 0x7FFFFFFF  # hash can return negative int
            with autocast_ctx:
                full_completions_group_mb, masks_group_mb = engine.generate_batch(
                    tokens, 
                    num_samples=args.device_batch_size, 
                    max_tokens=args.max_new_tokens, 
                    temperature=1.0, 
                    top_k=50, 
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
        rewards_group = torch.tensor(rewards_group)
        mu = rewards_group.mean()
        advantages_group = rewards_group - mu

        # add assistant end token because engine.generate does not.
        # TODO: Investigate why does engine not generate the the end token? What's the advantage?
        # TODO: Why should we not take a loss over the end token?
        #   Is it to prevent the model from getting stuck at never predicting end = unending reasoning = 0 reward = 0 loss
        #   Seems like the idea is: teach the model reasoning strength during SFT and only modify the value of the tokens (not number of tokens) during RL.
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
        targets = completions_tensor[:, 1: ].clone()  # clone to avoid in-place copy
        targets[masks_tensor[:, 1: ] == 0] = -1  # gpt.py sets ignore index to -1

        # send generations_group and rewards_group for logging only
        yield generations_group, inputs, targets, rewards_group, advantages_group

if __name__ == "__main__":
    gsm8k = GSM8K(subset="main", split="train")

    ddp_rank, ddp_local_rank, ddp_world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 0))
    if ddp_world_size > 1 and args.device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(args.device_type)  # "cpu|cuda"

    model, tokenizer, meta_data = load_model(source=args.source, phase="eval", step=args.step, device=device)
    engine = Engine(model, tokenizer)
    total_steps = (len(gsm8k) // args.examples_per_step) * args.epochs
    
    optimizers = model.setup_optimizers()
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"] * args.init_lr_frac

    def get_lrm(it, total_steps):
        return 1.0 - it / total_steps  # ramp down to 0
    
    n_examples_per_rank = args.examples_per_step // ddp_world_size
    dataset_iterator = functools.partial(get_rollout_group, model, gsm8k, engine, tokenizer, args, ddp_world_size)
    for step in range(total_steps):
        rewards_step = []
        lengths_step = []
        for example_idx in range(n_examples_per_rank):
            rollout_group = next(dataset_iterator(step))
            generations_group, inputs, targets, rewards_group, advantages_group = rollout_group
            model.train()
            n_microbatches = args.num_samples // args.device_batch_size
            for batch_start in range(0, args.num_samples, args.device_batch_size):
                inputs_batch = inputs[batch_start: batch_start + args.device_batch_size].to(device)
                targets_batch = targets[batch_start: batch_start + args.device_batch_size].to(device)
                advantages_batch = advantages_group[batch_start: batch_start + args.device_batch_size].to(device)

                with autocast_ctx:
                    logp_per_token = -model.forward(inputs_batch, targets_batch, loss_reduction="none")
                # ignore_index = -1 in model.forward will force the loss to be 0 for padded tokens
                logp_per_token = logp_per_token.view(args.device_batch_size, -1)  # (BS*T) -> (BS, T)
                advantages_batch = advantages_batch.unsqueeze(-1)  # (BS, 1)
                # No IS weight since we are on-policy
                assert advantages_batch.shape == (args.device_batch_size, 1), f"{advantages_batch.shape=} != {(args.device_batch_size, 1)}"
                pg_loss_per_token = logp_per_token * advantages_batch
                non_padded_tokens_per_sample = (inputs_batch != -1).sum(-1, keepdim=True)
                pg_loss_per_sample = pg_loss_per_token.sum(-1, keepdim=True) / non_padded_tokens_per_sample
                pg_loss = pg_loss_per_sample.sum() / (n_examples_per_rank * n_microbatches)
                pg_loss = -pg_loss  # maximize pi(trajectory) * adv(trajectory) => minimize loss
                pg_loss.backward()
                # no need to log advantages as their sum is always 0
                print0(f"Step: {step}/{total_steps} example: {example_idx}/{n_examples_per_rank} loss: {pg_loss.item():.4f} rewards mean: {rewards_group.mean().item():.4f}")
                rewards_step.append(rewards_group.mean().item())
                lengths_step.extend(len(seq) for seq in generations_group)
        
        # TODO: do this logging after reducing the rewards and gen lengths across ranks
        rewards_step_mean = sum(rewards_step) / len(rewards_step)
        generation_lengths_mean = sum(lengths_step) / len(lengths_step)
        print0(f"Step: {step}/{total_steps} Avg rewards: {rewards_step_mean} Avg gen len: {generation_lengths_mean}")
        if not args.no_wandb and ddp_rank == 0:
            wandb_run.log({
                "step": step,
                "reward": rewards_step_mean,
                "sequence_length": generation_lengths_mean,
                "lrm": get_lrm(step, total_steps)
            })
        
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lrm(step, total_steps)
            opt.step()
        model.zero_grad(set_to_none=True)
    wandb_run.finish()
    