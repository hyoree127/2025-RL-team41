# quantize_with_llmcompressor_dataset_c4.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
import torch
import glob
import os

# model_path = '../models/Llama-3.1-8B'
# quant_path = '../models/Llama-3.1-8B-awq-redpajama'

model_path = '../models/Llama-3-8B-Instruct'
quant_path = '../models/Llama-3-8B-Instruct-llmcomp4bit-c4'

os.makedirs(quant_path, exist_ok=True)

print("=" * 50)
print("LLM Compressor AWQ Quantization with RedPajama")
print("=" * 50)

# 1. 모델 로딩
print("\n[1/4] Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 데이터 로드
print("\n[2/4] Loading calibration data...")

data_files = glob.glob("../data/c4/*.jsonl")
if len(data_files) == 0:
    raise ValueError("❗ ../data/rp 아래에 .jsonl 파일 없음.")

calib_dataset = load_dataset(
    "json",
    data_files=data_files,
    split="train"
)
calib_dataset = calib_dataset.shuffle(seed=42)

# 3. 캘리브레이션 샘플 준비
print("\n[3/4] Preparing calibration samples...")
calib_samples = []
n_samples = 512
max_seq_len = 512

for sample in calib_dataset:
    if len(calib_samples) >= n_samples:
        break

    text = sample.get("text", "")
    if not isinstance(text, str):
        continue

    text = text.strip()
    if len(text) == 0:
        continue

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_seq_len or len(tokens) < 10:
        continue

    calib_samples.append(text)

    if len(calib_samples) % 100 == 0:
        print(f"  Collected {len(calib_samples)}/{n_samples} samples...")

print(f"✓ Prepared {len(calib_samples)} calibration samples")

calib_dataset_dict = Dataset.from_dict({"text": calib_samples})

print("Converted to HuggingFace Dataset format")

# 4. AWQ 실행 (핵심)
print("\n[4/4] Running AWQ quantization...")

recipe = [
    AWQModifier(
        scheme="W4A16",          # weight 4bit
        targets=["Linear"],      # Linear 레이어 전체
        ignore=[
            "lm_head",
            "embed_tokens",
            "re:.*norm.*"
        ],
    )
]

oneshot(
    model=model,
    dataset=calib_dataset_dict,
    recipe=recipe,
    max_seq_length=max_seq_len,
    num_calibration_samples=len(calib_samples),
    output_dir=quant_path,
)

print("\n✓ Quantization completed!")

tokenizer.save_pretrained(quant_path)
model.save_pretrained(quant_path, save_compressed=True) # model data저장인지 모르겠음
print("✓ Tokenizer saved.")
print("\nModel saved to:", quant_path)
