#!/bin/bash

#SBATCH --job-name=llm-evaluation_mistral_41 # Job name
#SBATCH --error=/home/daria.kotova/ai/llm-optimized-fintuning/logs/%j%x.err # error file
#SBATCH --output=/home/daria.kotova/ai/llm-optimized-fintuning/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l5-009


echo 'starting.......................'
###################### RUN LLM Finetune ######################

# MODEL_NAME='meta-llama/Llama-2-7b-hf'
MODEL_NAME='mistralai/Mistral-7B-v0.1'
WANDB_PROJECT=llm_evaluation

echo $WANDB_PROJECT

python calculate_perplexity.py \
--model_name=$MODEL_NAME \
--per_device_val_batch_size=1 \
--checkpoint_path='experiments/Mistral-7B-v0.1/checkpoint-41000'

# --checkpoint_path='experiments/Llama-2-7b-hf/checkpoint-41000' \

echo ' ending '