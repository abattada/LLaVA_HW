import os

# 建議先指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 如果之後換成 Qwen2.5-VL 這類，也可以順便設 MAX_PIXELS 之類的 env
# os.environ["MAX_PIXELS"] = "1003520"

from swift.llm import PtEngine, RequestConfig, InferRequest

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def build_engine():
    # 建一個推理引擎，4090 跑 7B 沒問題
    engine = PtEngine(MODEL_ID, max_batch_size=2)  # 每次最多 2 個 request 一起跑
    cfg = RequestConfig(
        max_tokens=512,   # 回答最長字數
        temperature=0.2,  # 隨機性，0～1 之間
    )
    return engine, cfg

def ask_on_image(engine, cfg, image_path, questions):
    """
    image_path: 圖片路徑 (str)
    questions: 針對這張圖的問題 list[str]
    """
    # 每個問題都當成一個獨立的 request，內容用 <image> 開頭
    reqs = []
    for q in questions:
        reqs.append(
            InferRequest(
                messages=[{"role": "user", "content": f"<image> {q}"}],
                images=[image_path],  # 綁定這張圖
            )
        )

    resp_list = engine.infer(reqs, request_config=cfg)
    answers = [r.choices[0].message.content for r in resp_list]
    return answers

def main():
    engine, cfg = build_engine()

    # 這裡填你 5 張圖的位置 & 每張要問的 2 個問題
    image_qas = {
        "/local/abat/LLaVA_HW/photo/image_08180.jpg": [
            "這張照片在拍什麼？",
            "這張照片裡你覺得最重要的資訊是什麼？",
        ],
        "/local/abat/LLaVA_HW/photo/image_08181.jpg": [
            "這張照片在拍什麼？",
            "有沒有什麼異常或值得注意的地方？",
        ],
        "/local/abat/LLaVA_HW/photo/image_08182.jpg": [
            "這張照片的場景大概在哪裡？",
            "你覺得這張照片的情緒是什麼？",
        ],
        "/local/abat/LLaVA_HW/photo/image_08183.jpg": [
            "照片中有哪些主要物體？",
            "這些物體之間有什麼關係？",
        ],
        "/local/abat/LLaVA_HW/photo/image_08184.jpg": [
            "這張照片大概是在做什麼活動？",
            "如果要幫這張照片取一個標題，你會怎麼取？",
        ],
    }

    for img, qs in image_qas.items():
        print("=" * 80)
        print(f"Image: {img}")
        answers = ask_on_image(engine, cfg, img, qs)
        for q, a in zip(qs, answers):
            print(f"\nQ: {q}")
            print(f"A: {a}")

if __name__ == "__main__":
    main()
