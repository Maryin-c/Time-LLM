#!/bin/bash

#SBATCH --job-name=deepseek_test
#SBATCH --output=./res/deepseek_long_term_ETTh1_336_res.txt
#SBATCH --error=./res/deepseek_long_term_ETTh1_336_error.txt
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../../miniconda3/etc/profile.d/conda.sh
conda activate qwen

model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

#master_port=25901
#num_process=1
batch_size=4
d_model=32
d_ff=128

comment='TimeLLM-ETTh1'

accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'COS'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model DeepSeek \
  --llm_dim 2048

