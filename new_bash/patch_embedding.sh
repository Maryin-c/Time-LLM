#!/bin/bash

#SBATCH --job-name=patch_embedding
#SBATCH --output=./res/patch_embedding_else_res.txt
#SBATCH --error=./res/patch_embedding_else_error.txt
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../../miniconda3/etc/profile.d/conda.sh
conda activate timellm

model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

#master_port=25000
#num_process=2
batch_size=8
d_model=32
d_ff=128

comment='TimeLLM-ETTh1'


##################################################################### MQA ###############
accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment


accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_720 \
  --model $model_name \
  --data ETTh1 \
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
  --train_epochs $train_epochs \
  --model_comment $comment 
##################################################################### MQA ###############












