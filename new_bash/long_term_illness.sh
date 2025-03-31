#!/bin/bash

#SBATCH --job-name=illness
#SBATCH --output=./res/illness_res.txt
#SBATCH --error=./res/illness_error.txt
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
batch_size=32
d_model=16
d_ff=32

comment='TimeLLM-illness'

accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness \
  --model $model_name \
  --data illness \
  --features M \
  --seq_len 70 \
  --label_len 7 \
  --pred_len 24 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1

accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness \
  --model $model_name \
  --data illness \
  --features M \
  --seq_len 70 \
  --label_len 7 \
  --pred_len 36 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1

accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness \
  --model $model_name \
  --data illness \
  --features M \
  --seq_len 70 \
  --label_len 7 \
  --pred_len 48 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1

accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness \
  --model $model_name \
  --data illness \
  --features M \
  --seq_len 70 \
  --label_len 7 \
  --pred_len 60 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1
