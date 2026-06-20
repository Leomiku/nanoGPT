import time


# FineWeb-Edu 20B-token pretraining config for GPT-2 small (~124M params).
# Launch on single 96GB GPU with:
# python train.py config/train_fineweb_edu_20b.py

out_dir = "out-fineweb-edu-20b-gpt2-small"
wandb_run_name = "fineweb-edu-20b-gpt2-small-" + str(int(time.time()))
wandb_log = True
wandb_project = "fineweb-edu"

dataset = "fineweb_edu_20b"

# 64 batch_size * 8 grad_accum/GPU * 1024 block_size = 528,288 tokens/iter
batch_size = 64
block_size = 1024
gradient_accumulation_steps = 8

# 20B tokens / 528,288 tokens per iter ~= 38,147 iterations.
max_iters = 38147
lr_decay_iters = 38147

# GPT-2 small / 124M parameter shape
n_layer = 12
n_head = 12
n_emdb = 768
dropout = 0.0
bias = False

learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warm_iters = 1000
min_lr = 6e-5

eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

device = "cuda"
dtype = "bfloat16"
compile = True