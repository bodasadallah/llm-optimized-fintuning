#!/bin/bash

#SBATCH --job-name=llm-evaluation_mistral_41 # Job name
#SBATCH --error=/home/daria.kotova/ai/llm-optimized-fintuning/logs/%j%x.err # error file
#SBATCH --output=/home/daria.kotova/ai/llm-optimized-fintuning/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --nodelist=ws-l6-009


echo "starting......................."
###################### RUN LLM Finetune ######################


# --model_name='mistralai/Mistral-7B-v0.1' \



# MODEL_NAME="meta-llama/Llama-2-7b-hf"
MODEL_NAME="mistralai/Mistral-7B-v0.1"

# --do_train \
WANDB_PROJECT=llm_evaluation

echo $WANDB_PROJECT

python calc_metrics.py \
--checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-41000" 
--per_device_val_batch_size=1

# --checkpoint_path="experiments/Llama-2-7b-hf/checkpoint-24000" \




echo " ending "
#srun python run_clm.py config.json