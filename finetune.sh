#!/bin/bash

#SBATCH --job-name=llm-finetuning # Job name
#SBATCH --error=/home/abdelrahman.sadallah/mbzuai/llm-optimized-fintuning/logs/%j%x.err # error file
#SBATCH --output=/home/abdelrahman.sadallah/mbzuai/llm-optimized-fintuning/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l5-001


echo "starting......................."
###################### RUN LLM Finetune ######################


WANDB_PROJECT=llm_finetuning

echo $WANDB_PROJECT
python train.py \
--checkpoint_path="experiments/checkpoint-400"


echo " ending "
#srun python run_clm.py config.json
