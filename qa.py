import os
import argparse
import json
from pathlib import Path

# 指定 GPU（看你環境需要，可以保留）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from swift.llm import PtEngine, RequestConfig, InferRequest
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

# ===== 路徑設定 =====
BASE_REMOTE_ID = "llava-hf/llava-1.5-7b-hf"
BASE_LOCAL_DIR = "./output/llava-1.5-7b-base"
FINETUNE_LOCAL_DIR = "./output/llava-vqarad-lora-swift/v4-20251123-225352"


def ensure_base_model():
    """確保 base 模型已經下載到 BASE_LOCAL_DIR，沒有就從遠端下載並儲存。"""
    base_dir = Path(BASE_LOCAL_DIR)
    config_path = base_dir / "config.json"

    if config_path.exists():
        print(f"[INFO] Found local base model at {BASE_LOCAL_DIR}")
        return BASE_LOCAL_DIR

    print(f"[INFO] Local base model not found at {BASE_LOCAL_DIR}")
    print(f"[INFO] Downloading base model from: {BASE_REMOTE_ID}")

    # 只在 CPU 上載入一次，然後存成 HF 格式
    model = LlavaForConditionalGeneration.from_pretrained(
        BASE_REMOTE_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    processor = AutoProcessor.from_pretrained(BASE_REMOTE_ID)

    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving base model to: {BASE_LOCAL_DIR}")
    model.save_pretrained(BASE_LOCAL_DIR)
    processor.save_pretrained(BASE_LOCAL_DIR)

    # 釋放暫時載入的模型（避免佔記憶體）
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BASE_LOCAL_DIR


def resolve_model_path(model_arg: str) -> str:
    """
    將使用者輸入的 --model 轉成實際要給 PtEngine 的路徑 / 模型 ID。
    - 'base'     -> 確保 ./output/llava-1.5-7b-base 存在，必要時自動下載
    - 'finetune' -> 檢查 FINETUNE_LOCAL_DIR 存在，否則回報錯誤
    - 其他       -> 原樣回傳（當成 hub id 或本地路徑）
    """
    if model_arg == "base":
        return ensure_base_model()

    if model_arg == "finetune":
        cfg_path = Path(FINETUNE_LOCAL_DIR) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"找不到 finetune 模型：{FINETUNE_LOCAL_DIR}\n"
                "請先完成微調並把合併後的模型存到這個路徑，"
                "或改用 --model base / --model <自訂模型路徑>。"
            )
        print(f"[INFO] Using finetuned model at {FINETUNE_LOCAL_DIR}")
        return FINETUNE_LOCAL_DIR

    # 其他情況一律當成已經是正確的 model id / 路徑
    return model_arg


def build_engine(model_path: str, max_batch_size: int = 2):
    """
    建立 PtEngine 推理引擎
    model_path:
        - 可以是 ModelScope / HF 的 model id
        - 也可以是你本地的 checkpoint 目錄
    """
    print(f"[INFO] Loading model from: {model_path}")
    engine = PtEngine(
        model_path,
        model_type="llava1_5_hf",
        max_batch_size=max_batch_size,
    )

    cfg = RequestConfig(
        max_tokens=512,   # 回答長度上限
        temperature=0.2,  # 隨機性
    )
    return engine, cfg


def ask_on_image(engine, cfg, image_path, questions):
    """
    image_path: 圖片路徑 (str)
    questions: 針對這張圖的問題 list[str]
    """
    reqs = []
    for q in questions:
        reqs.append(
            InferRequest(
                messages=[{"role": "user", "content": f"<image> {q}"}],
                images=[image_path],
            )
        )

    resp_list = engine.infer(reqs, request_config=cfg)
    answers = [r.choices[0].message.content for r in resp_list]
    return answers


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLaVA inference on custom photos (ms-swift PtEngine)."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help=(
            "模型 ID 或本地 checkpoint 路徑，或以下關鍵字：\n"
            "  base     -> 使用 ./output/llava-1.5-7b-base，若不存在會自動下載 llava-hf/llava-1.5-7b-hf 並儲存\n"
            "  finetune -> 使用 ./output/llava-vqarad-lora-swift/v4-20251123-225352，若不存在則報錯\n"
            "也可以直接給：\n"
            "  --model llava-hf/llava-1.5-7b-hf\n"
            "  --model ./output/llava-vqarad-lora-swift/xxx"
        ),
    )

    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=2,
        help="PtEngine 的 max_batch_size（一次最多並行處理多少問題）。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 將 base / finetune 轉成實際路徑，並處理下載 / 檢查
    try:
        resolved_model_path = resolve_model_path(args.model)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # 建立引擎，可以選 base / finetuned checkpoint / 其他自訂模型
    engine, cfg = build_engine(resolved_model_path, max_batch_size=args.max_batch_size)

    # 這裡填你 N 張圖的位置 & 每張要問的問題
    image_qas = {
        "/local/abat/LLaVA_HW/photo/cat.jpeg": [
            "這張照片中的動物是什麼？有幾個？",
            "這張照片可能是在哪裡拍的？",
        ],
        "/local/abat/LLaVA_HW/photo/elsa.jpeg": [
            "這張照片的場景大概在哪裡？",
            "你覺得這張照片的情緒是什麼？",
        ],
        "/local/abat/LLaVA_HW/photo/gate.jpeg": [
            "這張照片的場景大概在哪裡？",
            "照片中有哪些東西？",
        ],
        "/local/abat/LLaVA_HW/photo/temple.jpeg": [
            "這張照片大概是在做什麼活動？",
            "如果要幫這張照片取一個標題，你會怎麼取？",
        ],
        "/local/abat/LLaVA_HW/photo/plates.jpeg": [
            "照片中有哪些主要物體？",
            "這張照片的場景大概在哪裡？",
        ],
        "/local/abat/LLaVA_HW/photo/rainbow.jpeg": [
            "照片中主要在拍什麼？",
            "這張照片的氛圍如何？",
        ],
    }

    results = []

    for img, qs in image_qas.items():
        answers = ask_on_image(engine, cfg, img, qs)
        for q, a in zip(qs, answers):
            results.append({
                "image": img,
                "question": q,
                "answer": a,
            })

    # 確保 output/qa 存在
    out_dir = "output/qa"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "qa_" , args.model, ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] QA results saved to: {out_path}")


if __name__ == "__main__":
    main()
