#!/bin/bash

#SBATCH --job-name=zeroshot
#SBATCH --output=./res/zero_shot_720_res.txt
#SBATCH --error=./res/zero_shot_720_error.txt
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../../miniconda3/etc/profile.d/conda.sh
conda activate timellm

model_name=TimeLLM
learning_rate=0.01
llama_layers=16

#master_port=25000
#num_process=2
batch_size=4
d_model=32
d_ff=128

comment='TimeLLM-ETTh1_ETTh2'

accelerate launch --mixed_precision bf16 run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTh1.csv \
  --data_path ETTh2.csv \
  --model_id ETTh1_ETTh2_512_720 \
  --model $model_name \
  --data_pretrain ETTh1 \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
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
  --train_epochs 5 \
  --model_comment $comment \
  --prompt_domain 1
