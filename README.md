# LLaVA-1.5 on VQA-RAD – Fine-tuning & Evaluation Pipeline

This repo fine-tunes **LLaVA-1.5-7B** on the **VQA-RAD** dataset using **LoRA** (via ms-swift & Hugging Face) and evaluates with **BLEU**.  
It also includes a simple manual QA script for qualitative comparison (base vs. fine-tuned model).

---

## 1. Environment Setup

### 1.1. Create Conda environment

conda create -n swift python=3.10
conda activate swift

### 1.2. Install packages
pip install -r requirements.txt


## 2. Zero shot
python prep.py

<pre>
This will download the vqa dataset and store in ./data
</pre>

python qa.py --model base

<pre>
Get the answers of 5 images and 3 test samples from vqa, stored in ./output/qa
Read the answers in ./output/qa/qa_base.json
This will download the base model - LLaVA-1.5-7b , and store it in ./output/llava-1.5-7b-base
</pre>

## 3. Finetuning
./finetune.sh

<pre>
This will get a lora adapter checkpoints in ./output/llava-1.5-7b-finetune/...
</pre>

Copy the full or relative path of the lora checkpoints.

python merge_model.py --path path_of_lora_checkpoints

<pre>
This will combine base model and lora adapter, store the merged model in ./output/merged
</pre>

## 4. BLEU score
python eval.py --model base

python eval.py --model merged

## 5. Finetuned model result
python qa.py --model merged
<pre>
Get the answers of 5 images and 3 test samples from vqa, stored in ./output/qa
Read the answers in ./output/qa/qa_merged.json
</pre>

## 6. LLM as a judge

Prompt and copy manually.

### 6.1 Gemini Question prompt
<pre>
Give me the answer of each question, don't repeat my question, each answer has max length 128
</pre>

### 6.2 ChatGPT System prompt
<pre>
You are now an expert who evaluates the image recognition capabilities of other language models. You will have to consider these categories:
Spatial reasoning ( top, down, right, ...)
Object recognition ( detail, blocked object )
Semantic understanding ( meaning of objects )
World knowledge ( culture, famous things )
Reasoning ( logic, commonsense )
Consistency ( Same reference in different question )

You have to use the same standard every round. For each round, you will be presented with an image and the questions. I will then give you the answers of two models, you have to evaluate the answers based on the categories and give a overall score to each model. The score is in range 0.0 ~ 10.0.
</pre>

### 6.3 ChatGPT Question prompt
<pre>
What are the objects in the top-left and top-right of the shelf? 
What film or television work is the theme of this scene?

Model A:
The objects in the top-left and top-right of the shelf are two figurines of princesses.
The theme of this scene is inspired by the popular animated film \"Frozen.\" In the image, there are two dolls, one dressed as Elsa and the other as Anna, both standing on a bookshelf.

Model B:
Objects in the top corners: The top-left features a dark horse statue, and the top-right contains a metal helmet or piece of armor (possibly a suit of armor's headpiece).
Theme of the scene: This scene is themed around the Disney animated film Frozen, featuring the young sisters, Elsa and Anna, as statues.
</pre>

### 6.4 Prompt record
Stored in ./prompts