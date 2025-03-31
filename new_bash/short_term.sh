#!/bin/bash

#SBATCH --job-name=shortterm
#SBATCH --output=./res/short_term_res.txt
#SBATCH --error=./res/short_term_error.txt
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../../miniconda3/etc/profile.d/conda.sh
conda activate timellm

model_name=TimeLLM
train_epochs=1
learning_rate=0.001
llama_layers=32

#master_port=25000
#num_process=2
batch_size=8
d_model=8
d_ff=32

comment='TimeLLM-M4'

accelerate launch --mixed_precision bf16 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --llm_layers $llama_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len 1 \
  --stride 1 \
  --batch_size 4 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --loss 'SMAPE' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1

accelerate launch --mixed_precision bf16 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --llm_layers $llama_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len 1 \
  --stride 1 \
  --batch_size $batch_size \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --loss 'SMAPE' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1

accelerate launch --mixed_precision bf16 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --llm_layers $llama_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len 1 \
  --stride 1 \
  --batch_size 4 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --loss 'SMAPE' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1

accelerate launch --mixed_precision bf16 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --llm_layers $llama_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len 1 \
  --stride 1 \
  --batch_size 4 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --loss 'SMAPE' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1

accelerate launch --mixed_precision bf16 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --llm_layers $llama_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len 1 \
  --stride 1 \
  --batch_size 4 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --loss 'SMAPE' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1

accelerate launch --mixed_precision bf16 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --llm_layers $llama_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len 1 \
  --stride 1 \
  --batch_size $batch_size \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --loss 'SMAPE' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1