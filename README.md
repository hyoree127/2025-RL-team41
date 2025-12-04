# AutoMP-RL: Automated Mixed-Precision Quantization using Reinforcement Learning

## Overview
This project introduces a Reinforcement Learning (RL) based mixed-precision quantization framework to reduce memory usage of large language models (LLMs) while minimizing perplexity degradation.

Instead of applying a fixed bit-width (like AWQ),
our method learns the optimal bit-width for every linear layer (3, 4, or 8 bits) using:

- Activation-aware state
- Layer-wise bit actions
- PPL + memory saving–based reward
- Per-channel/group-wise quantization
- REINFORCE policy gradient algorithm

## Repository Structure
```
state.py                        # Activation statistics collector
policy.py                       # Policy network (layer-wise bit selector)
reward.py                       # Reward: PPL penalty + memory saving
trainer.py                      # REINFORCE training loop
quantizer.py                    # Per-channel/group quantization backend
visualizer.py                   # Plotting activation / weight statistics
main.py                         # Full training pipeline
quantize_with_llmcompressor...  # AWQ baseline implementation
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
| OS           | Ubuntu 20.04/22.04, WSL2 |4
LLaMA-3-8B requires at least 48GB GPU memory for FP16 weights.

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
The project uses:
Meta-Llama-3-8B-Instruct
Download via HuggingFace:
```
huggingface-cli download \
  meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir ./models/Llama-3-8B-Instruct \
  --include "pytorch_model*.bin" "tokenizer*" "config.json"
```
** Requires LLaMA access approval from Meta/HuggingFace. **

## Run RL Training Pipeline
Run the full mixed-precision RL loop:
```
python main.py
```
This performs:
- Load calibration + evaluation data
- Collect activation statistics (forward hook)
- Initialize policy + reward modules
- Compute baseline PPL
- Run REINFORCE over episodes
- Save best configurations
- Save quantized model weights

## System Design
**State**
Layer activation mean (absolute value) from forward hooks
→ Indicates quantization sensitivity

**Action**
Choose one bit from: {3, 4, 8}

**Reward**
 //////// 수식 ////////
- α = 0.1
- β = 2.0

**Quantizer**
- Per-channel or group-wise quantization
- Group size = 128
- Full weight restoration before re-quantizing
- Implemented in quantizer.py

**RL Algorithm**
REINFORCE (Policy Gradient) 
with entropy bonus + advantage normalization

## System Design
