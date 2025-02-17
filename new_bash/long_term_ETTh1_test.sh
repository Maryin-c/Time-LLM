#!/bin/bash

#SBATCH --job-name=time_llm_test
#SBATCH --output=job_output_vit.txt
#SBATCH --error=job_error_vit.txt
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../../miniconda3/etc/profile.d/conda.sh
conda activate timellm

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

#master_port=25000
#num_process=2
batch_size=8
d_model=32
d_ff=128

comment='TimeLLM-ETTh1'

#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
