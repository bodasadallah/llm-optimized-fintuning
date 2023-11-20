#!/bin/bash

#SBATCH --job-name=bert_base # Job name
#SBATCH --error=/home/anastasiia.demidova/llm/llm-optimized-fintuning/logs/%j%x.err # error file
#SBATCH --output=/home/anastasiia.demidova/llm/llm-optimized-fintuning/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l5-003


echo "starting......................."


MODEL_NAME="Llama-2-7b-hf"
# MODEL_NAME="jais"
# MODEL_NAME="Mistral-7B-v0.1"


echo $MODEL_NAME
python calc_metrics_bert_copy.py --model_name="meta-llama/Llama-2-7b-hf" 
# python calc_metrics_bert_copy.py --model_name="mistralai/Mistral-7B-v0.1"



echo " ending "

