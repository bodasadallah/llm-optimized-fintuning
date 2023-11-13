#!/bin/bash

#SBATCH --job-name=llm-finetuning # Job name
#SBATCH --error=/home/anastasiia.demidova/llm/llm-optimized-fintuning/logs/%j%x.err # error file
#SBATCH --output=/home/anastasiia.demidova/llm/llm-optimized-fintuning/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l5-006


echo "starting......................."
###################### RUN LLM Finetune ######################


# --model_name='mistralai/Mistral-7B-v0.1' \



MODEL_NAME="meta-llama/Llama-2-7b-hf"

# MODEL_NAME="/home/abdelrahman.sadallah/.cache/huggingface/hub/jais"
# MODEL_NAME="mistralai/Mistral-7B-v0.1"

# --do_train \
WANDB_PROJECT=llm_finetuning

echo $WANDB_PROJECT
python train.py \
--save_steps=1000 \
--eval_steps=10000 \
--do_eval=1 \
--report_to="all" \
--logging_steps=500 \
--logging_dir="experiments/$MODEL_NAME" \
--model_name=$MODEL_NAME \
--run_name=$MODEL_NAME \
--per_device_train_batch_size=4 \
--per_device_val_batch_size=4 \
--gradient_accumulation_steps=2 \
--gradient_checkpointing=1 \
--model="english" \
--checkpoint_path="experiments/Llama-2-7b-hf/checkpoint-24000"
# --use_flash_attention_2=1 \
# --lora_target_modules "c_attn" "c_proj"

# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 




echo " ending "
#srun python run_clm.py config.json