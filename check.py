from transformers import AutoProcessor, LlavaForConditionalGeneration

base_dir = "./output/llava-1.5-7b-base"
merged_dir = "./output/merged"

base = LlavaForConditionalGeneration.from_pretrained(base_dir)
merged = LlavaForConditionalGeneration.from_pretrained(merged_dir)

print(base.num_parameters(), merged.num_parameters())
