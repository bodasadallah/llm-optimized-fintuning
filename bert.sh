#!/bin/bash

#SBATCH --job-name=bert # Job name
#SBATCH --error=/home/anastasiia.demidova/llm/llm-optimized-fintuning/logs/%j%x.err # error file
#SBATCH --output=/home/anastasiia.demidova/llm/llm-optimized-fintuning/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l3-001


echo "starting......................."


# MODEL_NAME="Llama-2-7b-hf"
MODEL_NAME="/home/anastasiia.demidova/.cache/huggingface/hub/jais"
# MODEL_NAME="Mistral-7B-v0.1"


echo $MODEL_NAME
# python calc_metrics_bert.py --checkpoint_path="experiments/Llama-2-7b-hf/checkpoint-41000"
python calc_metrics_bert_arabic.py --checkpoint_path="experiments/jais/checkpoint-24000" --model_name=$MODEL_NAME
# # --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 


echo " ending "

