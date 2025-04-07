#!/bin/bash

#SBATCH --job-name=final_extend
#SBATCH --output=./res/final_extend_res.txt
#SBATCH --error=./res/final_extend_error.txt
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../../miniconda3/etc/profile.d/conda.sh
conda activate qwen

model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

#master_port=25000
#num_process=2
batch_size=32
d_model=16
d_ff=32

comment='TimeLLM-exchange_rate'

##################################################################### 64 ###############
accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate \
  --model $model_name \
  --data exchange_rate \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 168 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --window_size 64 \
  --llm_model DeepSeek \
  --llm_dim 2048 \
  --wandb_title 'long-term-extend-data' \
  --prompt_domain 1


accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate \
  --model $model_name \
  --data exchange_rate \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 672 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --window_size 64 \
  --llm_model DeepSeek \
  --llm_dim 2048 \
  --wandb_title 'long-term-extend-data' \
  --prompt_domain 1
##################################################################### 64 ###############
