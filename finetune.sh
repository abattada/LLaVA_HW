#!/bin/bash

#   --gradient_accumulation_steps 8 \
python -m swift.cli.sft \
  --model_type llava1_5_hf \
  --model ./output/llava-1.5-7b-base \
  --train_type lora \
  --dataset data/jsonl/train.jsonl \
  --num_train_epochs 10 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4 \
  --gradient_checkpointing true \
  --freeze_vit true \
  --lora_rank 8 \
  --lora_alpha 8\
  --torch_dtype bfloat16 \
  --output_dir output/llava-1.5-7b-finetune \
  --logging_steps 50 \
  --logging_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1  \
  --max_length 768
