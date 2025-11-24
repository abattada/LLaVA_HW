#!/bin/bash
# 建議前面有 activate，不過你已經在對的 env 裡執行就沒差
# conda activate /home/abat/conda_envs/swift

python -m swift.cli.sft \
  --model_type llava1_5_hf \
  --model ./output/llava-1.5-7b-base \
  --train_type lora \
  --dataset data/jsonl/train.jsonl \
  --num_train_epochs 30 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 \
  --gradient_checkpointing true \
  --freeze_vit true \
  --torch_dtype bfloat16 \
  --output_dir output/llava-vqarad-lora-swift \
  --logging_steps 200 \
  --logging_strategy epoch \
  --save_strategy no
