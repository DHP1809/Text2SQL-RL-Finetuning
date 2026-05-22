#!/bin/bash
source /workspace/.venv/bin/activate
cd /workspace/EasyR1
sudo apt-get update && sudo apt-get install python3.12-dev

#export RAY_DEBUG_POST_MORTEM=1
#export VLLM_LOGGING_LEVEL=DEBUG
#export VLLM_TRACE_FUNCTION=1          # very verbose, shows every function call
#export CUDA_LAUNCH_BLOCKING=1        # helps pinpoint CUDA kernel issues
export VLLM_USE_V1=0

set -x

MODEL_PATH="/model"  # replace it with your local file path #Qwen/Qwen3-Coder-30B-A3B-Instruct

CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main \
    config=/workspace/EasyR1/examples/config.yaml \
    data.train_files=/workspace/dataset/train_dataset_rl_6000.json \
    data.val_files=/workspace/dataset/train_dataset_rl_601.json \
    data.max_prompt_length=9216 \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=2 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.2 \
    worker.actor.clip_ratio_dual=10.0 \
    trainer.experiment_name=qwen3_coder_30b_a3b_text2sql_grpo \
    trainer.n_gpus_per_node=2