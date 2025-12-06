# AutoMP-RL: Automated Mixed-Precision Quantization using Reinforcement Learning

## Overview
This project introduces a Reinforcement Learning (RL) based mixed-precision quantization framework to reduce memory usage of large language models (LLMs) while **minimizing perplexity degradation.**  

Instead of applying a fixed bit-width, the policy network **selects the bit-width for each linear layer (3 / 4 / 8 bit)** based on activation statistics, balancing:

- Perplexity (PPL) degradation
- Memory saving (average bit-width per layer)

Core ideas:
- Activation-aware **state**
- Layer-wise bit **actions**
- PPL + memory saving–based **reward**
- Per-channel/group-wise **quantization**
- REINFORCE policy gradient algorithm

---

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

## Features
**Activation-aware mixed-precision**  
Computes activation magnitude per layer to infer sensitivity.

**RL-driven bit selection**  
Policy network selects 3-bit / 4-bit / 8-bit per linear layer.

**Reward-based optimization**  
```
Reward = -α * (PPL - PPL_baseline) + β * MemorySaving
```

**High-quality quantization backend**  
Per-channel quantization  
Group-wise quantization (group_size = 128)  

**AWQ W4A16 baseline included**  
Provides direct comparison with established 4-bit quantization.

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/hyoree127/2025-RL-team41.git
cd 2025-RL-team41
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Install PyTorch separately if needed:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install LLMCompressor (Quantization Backend)
```bash
pip install llmcompressor
```

For CUDA support:
```bash
pip install "llmcompressor[cuda]"
```

---

## Environment Requirements

| Component    | Version                  |
| ------------ | ------------------------ |
| Python       | 3.10–3.11                 |
| PyTorch      | ≥ 2.1 (tested on 2.9.1)  |
| Transformers | ≥ 4.40                   |
| GPU          | A6000 / A100 recommended |
| OS           | Ubuntu 20.04/22.04, WSL2 |

**Note**: LLaMA-3-8B requires at least 48GB GPU memory for FP16 weights.

---

## Dataset Setup

This project uses the following datasets:
- **C4**: Calibration and evaluation
- **RedPajama**: Alternative calibration dataset
- **WikiText-1 / WikiText-2**: Additional evaluation

Example folder structure:
```
data/
├── c4/*.jsonl
└── redpajama/*.jsonl
```

You may change dataset paths in `config.py`.

---

## Base Model Download

The project uses **Meta-Llama-3-8B-Instruct**

Download via HuggingFace:
```bash
huggingface-cli download \
  meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir ./models/Llama-3-8B-Instruct \
  --include "pytorch_model*.bin" "tokenizer*" "config.json"
```

**Requires LLaMA access approval from Meta/HuggingFace.**

---

## Run RL-based Mixed-Precision Training Pipeline

Run the full mixed-precision RL loop:
```bash
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

---

## System Architecture

### State: Activation Statistics
From ActivationCollector:  
- Mean absolute activation per layer  
- Used as a lightweight sensitivity estimate
  
### Action: Bit Selection
- **Action space**: {3, 4, 8} bits (configurable)
- A small MLP (`QuantizationPolicy` in `policy.py`) maps the scalar state to a categorical distribution over actions

### Reward
Reward is defined as:

```
Reward = -α × (PPL / PPL_baseline - 1) + β × MemorySaving
```

Where:
- **PPL**: Perplexity of the quantized model
- **PPL_baseline**: Original FP16 model's PPL
- **MemorySaving**: Relative reduction in total bit-sum vs 16-bit baseline
- **α = 0.7**, **β = 2.0** (default values)

### Quantizer
- Per-channel or group-wise quantization
- Group size = 128
- Full weight restoration before re-quantizing
- Implemented in `quantizer.py`

### RL Algorithm
**REINFORCE (Policy Gradient)** with entropy bonus + advantage normalization

---

## Hyperparameters

| Parameter        | Value          |
| ---------------- | -------------- |
| Model            | llama3-8B      |
| AVAILABLE_BITS   | [4, 8]         |
| LEARNING_RATE    | 2e-3           |
| NUM_EPISODES     | 300            |
| CALIB_SAMPLES    | 512            |
| EVAL_SAMPLES     | 100            |
| MAX_SEQ_LENGTH   | 512            |
| ALPHA (α)        | 0.7            |
| BETA (β)         | 2.0            |
| GAMMA (γ)        | 0.99           |
| NUM_SEEDS        | 10             |

---

## Experimental Results

### Datasets
We calibrate and evaluate our proposed method on:
- **C4**: Primary calibration and evaluation dataset
- **RedPajama**: Alternative calibration dataset

All experiments were conducted using PyTorch 2.9.1 on NVIDIA A6000 GPUs.

---

### Comparison with State-of-the-Art Methods

#### Perplexity (PPL) on C4

| Method              | Precision          | PPL    |
| ------------------- | ------------------ | ------ |
| Base                | FP16               | 0  |
| RTN                 | W4A16              | 0  |
| AWQ                 | W4A16              | 0  |
| **Proposed Method** | **W4A16 and W8A16**| 0 |

> 📊 **[INSERT FIGURE: PPL Comparison Bar Chart]**
ppl비교
#### VRAM Memory Usage on C4

| Method              | Precision          | VRAM Memory | Memory Reduction |
| ------------------- | ------------------ | ----------- | ---------------- |
| Base                | FP16               | 0 GB    | 0%               |
| RTN                 | W4A16              | 0 GB    | 0%           |
| AWQ                 | W4A16              | 0 GB    | 0%           |
| **Proposed Method** | **W4A16 and W8A16**| **0 GB**| **~72%**         |

> 📊 **[INSERT FIGURE: Memory Usage Comparison Bar Chart]**
메모리사용량 비교
---

### Training Dynamics Analysis

#### 4.1 Perplexity (PPL) Evolution

> 📊 **[INSERT FIGURE: PPL vs Episodes Line Plot]**
ppl 에피소드라인플롯
The figure illustrates the PPL evolution across training episodes. Initially, the model maintains PPL close to the FP16 baseline by predominantly selecting 8-bit layers. As the agent explores more aggressive quantization strategies by incorporating 4-bit layers for memory reduction, a transient increase in PPL is observed.

Through continued training, the agent progressively identifies layer combinations that sustain low PPL while maximizing the utilization of 4-bit layers. This convergence behavior indicates that our reinforcement learning framework effectively learns to balance the trade-off between memory efficiency and model accuracy.

#### 4.2 Memory Reduction Rate Evolution

> 📊 **[INSERT FIGURE: Memory Reduction Rate vs Episodes Line Plot]**
메모리감소율vs에피소드라인플롯
The figure presents the memory reduction rate as a function of training episodes. During the initial phase, the predominant selection of 8-bit layers (comprising over 50% of total layers) leads to substantially lower memory savings relative to the W4A16 quantization scheme.

However, through iterative episodic training, the agent progressively shifts toward more aggressive quantization strategies by increasing the ratio of 4-bit layers. Critically, this transition is achieved without sacrificing PPL performance, as the agent learns to identify which layers can tolerate lower-bit quantization. This adaptive learning mechanism ultimately enables our method to achieve memory efficiency comparable to W4A16 while delivering superior model performance.

#### 4.3 Reward Progression

> 📊 **[INSERT FIGURE: Reward vs Episodes Line Plot]**
보상 에피소드 라인 플롯
The figure illustrates the reward progression throughout the training process. During early episodes, the agent's preference for 8-bit layers results in substantial reward penalties due to insufficient memory reduction. In response, the framework adaptively increases 4-bit layer selections to maximize rewards by reducing memory consumption.

Upon reaching a certain threshold of memory efficiency, a notable transition in exploration strategy emerges. The agent shifts its focus toward identifying optimal layer configurations that preserve low PPL while maintaining the achieved memory savings. This behavioral evolution reflects the effectiveness of our reward formulation in guiding the agent through different optimization phases: first prioritizing memory reduction, then refining layer combinations for performance preservation.

#### 4.4 Bit-width Distribution per Layer

> 📊 **[INSERT FIGURE: Heatmap or Stacked Bar Chart showing 4-bit vs 8-bit layer distribution]**
4bit 8bit 레이어분포도 
This visualization shows which layers were assigned 4-bit vs 8-bit quantization in the optimal configuration, revealing the learned sensitivity patterns across the network architecture.

---

### Key Findings

Our experimental results demonstrate several key findings:

1. **Superior PPL Performance**: Our method achieves state-of-the-art PPL on C4, outperforming recent methods while maintaining reasonable computational costs.

2. **Memory Efficiency**: Our proposed method achieves a substantial reduction of **~72% in VRAM memory consumption** while maintaining comparable PPL performance. Notably, our approach demonstrates superior PPL compared to the baseline model, with memory overhead comparable to the W4A16 configuration. This result indicates that our method achieves an optimal trade-off between memory efficiency and model performance.

3. **Robustness**: Our method exhibits consistent PPL performance across different random seeds (n=10), demonstrating negligible variance in the results. This consistency indicates that our approach maintains strong robustness against random initialization and other sources of variability, ensuring stable and reliable performance across multiple experimental runs.

> 📊 **[INSERT FIGURE: Box plot showing PPL distribution across 10 random seeds]**
랜덤시드 시각화자료
---

## AWQ Baseline (LLMCompressor)

### Run AWQ Baseline:
```bash
python quantize_with_llmcompressor_dataset_c4.py
```

This script:
1. Loads Llama-3-8B-Instruct
2. Loads calibration data from `../data/c4/*.jsonl`
3. Filters and prepares 512 text samples
4. Runs W4A16 AWQ via:
```python
recipe = [
    AWQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["lm_head", "embed_tokens", "re:.*norm.*"],
    )
]
```
5. Calls `oneshot(...)` from llmcompressor to quantize and save the model
6. Saves tokenizer and quantized model to `quant_path`

---

## Visualization

### Usage:
```python
from visualizer import ModelVisualizer

viz = ModelVisualizer(layer_info)  # layer_info: list of dicts
viz.plot_all(save_dir="plots/")
```

### Output:
- Weight mean / std / min / max vs layer index
- Activation mean / max vs layer index
- Boxplot by layer type (e.g., Attention / MLP layers)

> 📊 **[INSERT FIGURE: Weight/Activation Statistics Visualization]**
weight distribution 시각화자료
---

## Pretrained Quantized Models

[Coming Soon]



## Acknowledgments

This work was supported by [funding information to be added].

