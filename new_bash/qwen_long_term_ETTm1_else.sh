#!/bin/bash

#SBATCH --job-name=qwen_test
#SBATCH --output=./res/qwen_long_term_ETTm1_else_res.txt
#SBATCH --error=./res/qwen_long_term_ETTm1_else_error.txt
#SBATCH --time=96:00:00
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

#master_port=25000
#num_process=2
batch_size=4
d_model=32
d_ff=128

comment='TimeLLM-ETTm1'

#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_512_96 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 96 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --lradj 'TST'\
#   --learning_rate 0.001 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment \
#   --llm_model Qwen \
#   --llm_dim 3584


# accelerate launch --mixed_precision bf16 run_main.py \
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./dataset/ETT-small/ \
  # --data_path ETTm1.csv \
  # --model_id ETTm1_512_192 \
  # --model $model_name \
  # --data ETTm1 \
  # --features M \
  # --seq_len 512 \
  # --label_len 48 \
  # --pred_len 192 \
  # --factor 3 \
  # --enc_in 7 \
  # --dec_in 7 \
  # --c_out 7 \
  # --des 'Exp' \
  # --itr 1 \
  # --d_model $d_model \
  # --d_ff $d_ff \
  # --batch_size $batch_size \
  # --learning_rate $learning_rate \
  # --lradj 'TST'\
  # --learning_rate 0.001 \
  # --llm_layers $llama_layers \
  # --train_epochs $train_epochs \
  # --patience 20 \
  # --model_comment $comment \
  # --llm_model Qwen \
  # --llm_dim 3584


accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_336 \
  --model $model_name \
  --data ETTm1 \
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
  --learning_rate $learning_rate \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --patience 20 \
  --model_comment $comment \
  --llm_model Qwen \
  --llm_dim 3584


accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_720 \
  --model $model_name \
  --data ETTm1 \
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
  --lradj 'TST'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --patience 20 \
  --model_comment $comment \
  --llm_model Qwen \
  --llm_dim 3584
