# AutoMP-RL: Automated Mixed-Precision Quantization using Reinforcement Learning

## Overview
This project introduces a Reinforcement Learning (RL) based mixed-precision quantization framework to reduce memory usage of large language models (LLMs) while minimizing perplexity degradation.

Instead of applying a fixed bit-width,  
the policy network **selects the bit-width for each linear layer (3 / 4 / 8 bit)** based on activation statistics,  

balancing:  

- Perplexity (PPL) degradation  
- Memory saving (average bit-width per layer)

Core ideas:

- Activation-aware **state**
- Layer-wise bit **actions**
- PPL + memory saving–based **reward**
- Per-channel/group-wise **quantization**
- REINFORCE policy gradient algorithm

## Repository Structure
```
main.py                         # Full RL training pipeline
state.py                        # Activation statistics collector
policy.py                       # Policy network for bit selection
reward.py                       # Reward: PPL penalty + memory saving
trainer.py                      # REINFORCE training loop
quantizer.py                    # Per-channel/group quantization backend
visualizer.py                   # Plotting activation / weight statistics
quantize_with_redpajama.py      # AWQ baseline implementation
config.py                       # Hyperparameters & global settings
```

## Installation
1. Clone Repository
```
git clone https://github.com/hyoree127/2025-RL-team41.git
cd 2025-RL-team41
```

2. Install Dependencies
```
pip install -r requirements.txt
```
- Install PyTorch separately if needed:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

3. Install LLMCompressor (Quantization Backend)
```
pip install llmcompressor
```
- CUDA version:
```
pip install "llmcompressor[cuda]"
```
## Environment Requirements
| Component    | Version                  |
| ------------ | ------------------------ |
| Python       | 3.9–3.11                 |
| PyTorch      | ≥ 2.1                    |
| Transformers | ≥ 4.40                   |
| GPU          | A6000 / A100 recommended |
| OS           | Ubuntu 20.04/22.04, WSL2 |  

- LLaMA-3-8B requires at least 48GB GPU memory for FP16 weights.

## Dataset Setup
This project uses the following datasets:  
- C4
- RedPajama
- WikiText-1 / WikiText-2

Example folder structure:
```
data/c4/*.jsonl
data/redpajama/*.jsonl
data/wikitext/*.jsonl
```
You may change dataset paths in config.py.

## Base Model Download
The project uses **Meta-Llama-3-8B-Instruct**  

Download via HuggingFace:  
```
huggingface-cli download \
  meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir ./models/Llama-3-8B-Instruct \
  --include "pytorch_model*.bin" "tokenizer*" "config.json"
```
**Requires LLaMA access approval from Meta/HuggingFace.**

## Run RL-based Mixed-Precision Training Pipeline
Run the full mixed-precision RL loop:
```
python main.py
```

This performs:
- Load calibration + evaluation datasets
- Collect activation statistics with ActivationCollector (forward hook)
- Initialize policy + reward modules
- Compute baseline PPL of the original model
- Run REINFORCE over episodes
- Track Best Reward and Best PPL configurations
- Save quantized model weights

## System Design
**State**
Layer activation mean (absolute value) from forward hooks
→ Indicates quantization sensitivity

**Action (Bit Selection)**
- Action space: {3, 4, 8} bits
- A small MLP (QuantizationPolicy in policy.py) maps the scalar state to a categorical distribution over actions.

**Reward**  
Reward is defined as:  
 //////// 수식 ////////  
 
- PPL : Perplexity of the quantized model
- PPL_baseline : Original FP16 model’s PPL
- MemorySaving : Relative reduction in total bit-sum vs 16-bit baseline
-  α = 0.1, β = 2.0
  
**Quantizer**
- Per-channel or group-wise quantization
- Group size = 128
- Full weight restoration before re-quantizing
- Implemented in quantizer.py

**RL Algorithm**  
REINFORCE (Policy Gradient) 
with entropy bonus + advantage normalization

## Hyperparameters
| Name           | Value     |
| -------------- | --------- |
| AVAILABLE_BITS | [3, 4, 8] |
| LEARNING_RATE  | 1e-3      |
| NUM_EPISODES   | 50        |
| CALIB_SAMPLES  | 512       |
| EVAL_SAMPLES   | 100       |
| MAX_SEQ_LENGTH | 512       |
| ALPHA          | 0.1       |
| BETA           | 2.0       |  

## Output

## AWQ Baseline(LLMCompressor)

- Run:
```
python quantize_with_llmcompressor_dataset_c4.py
```

- This script:
1. Loads Llama-3-8B-Instruct
2. Loads calibration data from ../data/c4/*.jsonl
3. Filters and prepares 512 text samples
4. Runs W4A16 AWQ via:
```
recipe = [
    AWQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["lm_head", "embed_tokens", "re:.*norm.*"],
    )
]
```
5. Calls oneshot(...) from llmcompressor to quantize and save the model
6. Saves tokenizer and quantized model to quant_path

## Visualization
```
from visualizer import ModelVisualizer

viz = ModelVisualizer(layer_info)  # layer_info: list of dicts
viz.plot_all(save_dir="plots/")
```

Output:  
- Weight mean / std / min / max vs layer index
- Activation mean / max vs layer index
- Boxplot by layer type (e.g., Attention / MLP layers)


## Pretrained Quantized Models
/////// 링크 추가 ///////
