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

# 所有 finetune / merged 模型統一放在這個資料夾底下
# 例如： ./output/llava-1.5-7b-finetune/v1
#       ./output/llava-1.5-7b-finetune/v4-20251123-225352
MERGED_LOCAL_DIR = "./output/merged"


def ensure_base_model() -> str:
    """確保 base 模型已經下載到 BASE_LOCAL_DIR，沒有就從遠端下載並儲存。"""
    base_dir = Path(BASE_LOCAL_DIR)
    config_path = base_dir / "config.json"

    if config_path.exists():
        print(f"[INFO] Found local base model at {BASE_LOCAL_DIR}")
        return BASE_LOCAL_DIR

    print(f"[INFO] Local base model not found at {BASE_LOCAL_DIR}")
    print(f"[INFO] Downloading base model from: {BASE_REMOTE_ID}")

    model = LlavaForConditionalGeneration.from_pretrained(
        BASE_REMOTE_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    processor = AutoProcessor.from_pretrained(BASE_REMOTE_ID)

    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving base model to: {BASE_LOCAL_DIR}")
    model.save_pretrained(BASE_LOCAL_DIR)
    processor.save_pretrained(BASE_LOCAL_DIR)

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BASE_LOCAL_DIR


def resolve_model_path(model_arg: str) -> str:
    """
    將 --model 轉成實際 path：

    - "base"         -> ./output/llava-1.5-7b-base（必要時自動下載）
    - 其他字串       -> 只在 ./output/llava-1.5-7b-finetune/<model_arg> 底下找
                       找不到就丟錯誤，不會再去 HF / ModelScope 下載。
    """
    if model_arg == "base":
        return ensure_base_model()
    elif model_arg == "merged":
        # 特例：merged 代表直接用 MERGED_LOCAL_DIR
        if Path(MERGED_LOCAL_DIR).is_dir():
            print(f"[INFO] Using merged model at {MERGED_LOCAL_DIR}")
            return str(Path(MERGED_LOCAL_DIR).resolve())
        else:
            raise FileNotFoundError(
                f"[ERROR] 找不到 merged 模型資料夾：{MERGED_LOCAL_DIR}\n"
                f"請確認你是否已經把 merged 模型存到這個目錄下。"
            )
    else:
        raise AttributeError(
            f"[ERROR] 參數不對\n"
        )


def build_engine(model_path: str, max_batch_size: int = 2):
    """
    建立 PtEngine 推理引擎
    model_path:
        - ./output/llava-1.5-7b-base
        - ./output/llava-1.5-7b-finetune/<某個子資料夾>
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
            "要使用的模型：\n"
            "  base         -> ./output/llava-1.5-7b-base（必要時自動下載 HF base）\n"
            "  merged       -> ./output/merged\n"
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

    # 把 base / 某個 finetune 名稱 轉成實際路徑，並處理下載 / 檢查
    try:
        resolved_model_path = resolve_model_path(args.model)
    except FileNotFoundError as e:
        print(e)
        return

    # 建立引擎，可以選 base / finetuned checkpoint
    engine, cfg = build_engine(resolved_model_path, max_batch_size=args.max_batch_size)

    # 這裡填你 N 張圖的位置 & 每張要問的問題
    image_qas = {
        "./photo/elsa.jpeg": [
            "Where are the objects in the top-left and top-right of the shelf?",
            "What film or television work is the theme of this scene?",
        ],
        "./photo/gate.jpeg": [
            "Where is the scene in this photo likely located?",
            "How is the weather in this photo?",
        ],
        "./photo/temple.jpeg": [
            "What do you see in this photo?",
            "What is the main emotion in this photo?",
        ],
        "./photo/plates.jpeg": [
            "Where is the most important feature in this store?",
            "What are the names of the tablewares on the shelf?",
        ],
        "./photo/rainbow.jpeg": [
            "What is the main object in this photo?",
            "What do you see besides the main object?",
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

    # 存成一個跟 model 名稱綁在一起的檔案
    out_path = os.path.join(out_dir, f"qa_{args.model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] QA results saved to: {out_path}")


if __name__ == "__main__":
    main()
