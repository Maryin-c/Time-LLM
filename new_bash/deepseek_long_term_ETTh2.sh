#!/bin/bash

#SBATCH --job-name=deepseek_test
#SBATCH --output=./res/deepseek_long_term_ETTh2_res.txt
#SBATCH --error=./res/deepseek_long_term_ETTh2_error.txt
#SBATCH --time=120:00:00
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

comment='TimeLLM-ETTh2'

#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_512_96 \
#   --model $model_name \
#   --data ETTh2 \
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
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment


accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_192 \
  --model $model_name \
  --data ETTh2 \
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
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate 0.002 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model DeepSeek \
  --llm_dim 2048



# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_512_336 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 336 \
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
#   --learning_rate 0.005 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment


accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_720 \
  --model $model_name \
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
  --d_model 16 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate 0.005 \
  --lradj 'TST'\
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --patience 10 \
  --model_comment $comment \
  --llm_model DeepSeek \
  --llm_dim 2048

