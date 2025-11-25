import os
import json
import re
import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from PIL import Image
import numpy as np

from transformers import AutoProcessor, LlavaForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ===== 固定資料路徑 =====
IMAGE_ROOT = "./data/images/test"      # 圖片資料夾
DATA_JSON_PATH = "./data/jsonl/test.jsonl"  # 問答 jsonl 檔

MAX_NEW_TOKENS_DEFAULT = 32

# ===== base / merged 模型設定 =====
BASE_REMOTE_ID = "llava-hf/llava-1.5-7b-hf"
BASE_LOCAL_DIR = "./output/llava-1.5-7b-base"
MERGED_LOCAL_DIR = "./output/merged"


# ===== 文本處理 & BLEU =====
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def compute_bleu_scores(preds, gts):
    smoothie = SmoothingFunction().method4
    scores = []
    for p, g in zip(preds, gts):
        p_n = normalize_text(p)
        g_n = normalize_text(g)
        if not g_n.strip():
            continue
        score = sentence_bleu(
            [g_n.split()],
            p_n.split(),
            weights=(0.5, 0.5),
            smoothing_function=smoothie,
        )
        scores.append(score)
    if not scores:
        return 0.0, 0.0, 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def extract_qa(item):
    """
    支援兩種格式：

    1) ms-swift style `messages`：
       a) content 是字串：
          {
            "messages": [
              {"role":"user","content":"<image> Question: ..."},
              {"role":"assistant","content":"答案文字"}
            ]
          }

       b) content 是 list：
          {
            "messages": [
              {
                "role":"user",
                "content":[{"type":"image"},{"type":"text","text":"Question: ..."}]
              },
              {
                "role":"assistant",
                "content":[{"type":"text","text":"答案文字"}]
              }
            ]
          }

    2) 舊的 `conversations` 格式 (保留原本邏輯)
    """
    # ----- 情況 1: messages -----
    if "messages" in item:
        msgs = item["messages"]
        if len(msgs) < 2:
            raise ValueError("messages 長度不足 2（需要 [user, assistant]）")

        user = msgs[0]
        assistant = msgs[1]

        # 解析 question
        u_content = user.get("content", "")
        if isinstance(u_content, str):
            # 整包都是字串：例如 "<image>\nQuestion: xxx"
            txt = u_content
            if "<image>" in txt:
                q = txt.split("<image>")[-1].strip()
            else:
                q = txt.strip()
        elif isinstance(u_content, list):
            # list 裡面可能是 dict 或 str
            q_parts = []
            for c in u_content:
                if isinstance(c, dict) and c.get("type") == "text":
                    q_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    q_parts.append(c)
            q = " ".join(q_parts).strip()
        else:
            # 其它型別就直接轉字串當作問題
            q = str(u_content).strip()

        # 解析 ground truth answer
        a_content = assistant.get("content", "")
        if isinstance(a_content, str):
            gt = a_content.strip()
        elif isinstance(a_content, list) and len(a_content) > 0:
            first = a_content[0]
            if isinstance(first, dict):
                gt = first.get("text", "").strip()
            else:
                gt = str(first).strip()
        else:
            gt = ""

        return q, gt

    # ----- 情況 2: conversations (舊格式) -----
    if "conversations" in item:
        convs = item["conversations"]
        if len(convs) < 2:
            raise ValueError("conversations 長度不足 2（需要 [human, gpt]）")
        human = convs[0]["value"]
        gt = convs[1]["value"].strip()
        if "<image>" in human:
            q = human.split("<image>")[-1].strip()
        else:
            q = human.strip()
        return q, gt

    raise ValueError("Unknown data format: no 'messages' or 'conversations'.")



# ===== 從 item 抽圖片路徑（固定在 IMAGE_ROOT） =====
def get_image_path(item) -> str:
    """
    讀 item 裡的 image / images 欄位，然後固定接在 IMAGE_ROOT 底下。

    - 如果是 item["image"] = "images/test/00001.jpg"
      -> 會取 basename "00001.jpg"，最後路徑是 ./data/images/test/00001.jpg

    - 如果是 item["images"] = ["images/test/00001.jpg"]
      -> 一樣取第一個再取 basename
    """
    if "image" in item:
        rel = item["image"]
    elif "images" in item:
        imgs = item["images"]
        if isinstance(imgs, list):
            rel = imgs[0]
        else:
            rel = imgs
    else:
        raise ValueError("Unknown data format: no 'image' or 'images'.")

    filename = os.path.basename(rel)
    full_path = os.path.join(IMAGE_ROOT, filename)
    return full_path


# ===== 讀固定路徑的 json/jsonl =====
def load_dataset():
    """
    使用固定的 DATA_JSON_PATH：
      ./data/jsonl/test.jsonl

    若為 .jsonl：每行一個 JSON
    若為 .json ：整體是一個 list
    """
    path = DATA_JSON_PATH
    data = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data


# ===== base / merged 模型處理 =====
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
    將 --model 轉成實際 path / id：

    - "base"         -> ./output/llava-1.5-7b-base（必要時自動下載）
    - 已存在的資料夾 -> 直接使用（適合 merged 模型）
    - 其他字串       -> 先找 ./output/merged/<model_arg>，有就用；
                       否則當成 HF / ModelScope 模型 ID
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
                f"找不到 merged 模型目錄：{MERGED_LOCAL_DIR}\n"
                f"請確認已經有合併好的模型存在這個資料夾。"
            )
    else:
        raise AttributeError(
            f"[ERROR] 參數不對\n"
        )



# ===== 載入模型 =====
def load_model_and_processor(model_path: str):
    print(f"[INFO] Loading processor & model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print("[INFO] Model loaded on", device)
    return model, processor, device


# ===== CLI 參數 =====
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on VQA-RAD test set with BLEU-2.")

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help=(
            "要 evaluate 的模型：\n"
            "  base          -> 使用 ./output/llava-1.5-7b-base（必要時自動下載）\n"
            "  merged        -> 使用 ./output/merged\n"
        ),
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS_DEFAULT,
        help="每次生成的最大 token 數（預設 32）。",
    )

    return parser.parse_args()


# ===== main =====
def main():
    args = parse_args()

    data = load_dataset()
    model_path = resolve_model_path(args.model)
    model, processor, device = load_model_and_processor(model_path)

    all_preds = []
    all_gts = []

    desc = f"Evaluating on {os.path.basename(DATA_JSON_PATH)}"
    for item in tqdm(data, desc=desc):
        img_path = get_image_path(item)
        image = Image.open(img_path).convert("RGB")

        q, gt = extract_qa(item)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        pred = processor.decode(gen_ids, skip_special_tokens=True)
        all_preds.append(pred)
        all_gts.append(gt)

    mean_bleu, std_bleu, n = compute_bleu_scores(all_preds, all_gts)

    print("======== VQA-RAD TEST BLEU-2 ========")
    print(f"Model path/id : {model_path}")
    print(f"Image root    : {IMAGE_ROOT}")
    print(f"Data jsonl    : {DATA_JSON_PATH}")
    print(f"Samples       : {n}")
    print(f"Mean BLEU-2   : {mean_bleu:.4f}")
    print(f"Std  BLEU-2   : {std_bleu:.4f}")


if __name__ == "__main__":
    main()
