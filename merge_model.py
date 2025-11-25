#!/usr/bin/env python
# merge_model.py
#
# 功能：
#   把 base LLaVA 模型 (./output/llava-1.5-7b-base)
#   和某個 LoRA checkpoint (透過 --path 傳入) 合併，
#   並把合併後的完整權重「直接」存到 ./output/merged 目錄（不再建立子資料夾）。
#
# 使用範例：
#   python merge_model.py --path output/llava-1.5-7b-finetune/v1
#   python merge_model.py --path output/llava-1.5-7b-finetune/v1 --merged_dir ./output/my_merged
#
# 注意：
#   每次執行都會覆蓋 merged_dir 裡原本的模型檔。

import os
import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


# 預設 base model 位置與 merged 目錄
DEFAULT_BASE_DIR = "./output/llava-1.5-7b-base"
DEFAULT_MERGED_DIR = "./output/merged"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge base LLaVA model with a LoRA adapter and save to ./output/merged."
    )

    parser.add_argument(
        "--base",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base 模型目錄（預設：{DEFAULT_BASE_DIR}）",
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="LoRA checkpoint 目錄，例如：output/llava-1.5-7b-finetune/v1",
    )

    parser.add_argument(
        "--merged_dir",
        type=str,
        default=DEFAULT_MERGED_DIR,
        help=f"合併後模型直接存放的目錄（預設：{DEFAULT_MERGED_DIR}）",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="載入模型時使用的 dtype（預設：float16）。",
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def main():
    args = parse_args()

    base_dir = Path(args.base)
    lora_dir = Path(args.path)
    merged_dir = Path(args.merged_dir)

    # ===== 檢查路徑 =====
    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"找不到 base 模型的 config.json：{base_dir}\n"
            f"請確認已經把 base LLaVA 模型存成 HF 格式到這個資料夾。"
        )

    if not lora_dir.is_dir():
        raise FileNotFoundError(
            f"找不到 LoRA 目錄：{lora_dir}\n"
            f"請確認 --path 有指到 ms-swift / peft 產生的 LoRA checkpoint。"
        )

    print(f"[INFO] Base model dir   : {base_dir}")
    print(f"[INFO] LoRA adapter dir : {lora_dir}")
    print(f"[INFO] Merged output dir: {merged_dir}")

    merged_dir.mkdir(parents=True, exist_ok=True)

    # ===== 載入 base 模型與處理器 =====
    dtype = get_torch_dtype(args.dtype)
    print(f"[INFO] Loading base LlavaForConditionalGeneration (dtype={args.dtype})...")

    model = LlavaForConditionalGeneration.from_pretrained(
        base_dir,
        torch_dtype=dtype,
        device_map="auto",   # 4090 可以直接 auto
    )

    processor = AutoProcessor.from_pretrained(base_dir)

    # ===== 把 LoRA 掛上去並合併 =====
    print(f"[INFO] Loading LoRA adapter from: {lora_dir}")
    model = PeftModel.from_pretrained(model, lora_dir)

    print("[INFO] Merging LoRA into base model (this may take a while)...")
    model = model.merge_and_unload()  # 只改記憶體中的 model，不會刪 LoRA 資料夾

    # ===== 儲存合併後模型 =====
    print(f"[INFO] Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)

    print("[INFO] Done. 你現在可以用這個 merged 目錄拿去 eval.py / qa.py 做推理或 BLEU 評估。")


if __name__ == "__main__":
    main()
