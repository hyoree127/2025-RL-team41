# quantizer.py
import gc
import torch
import torch.nn as nn
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from typing import Dict

def quantize_weight_per_channel(weight: torch.Tensor, n_bit: int) -> torch.Tensor:
    """
    Channel별로 scale을 계산하되, 모든 channel을 동일한 bit로 quantization (Vectorized Version)
    
    Args:
        weight: [out_channels, in_channels]
        n_bit: layer 전체에 적용할 bit 수 (예: 4, 8)
    
    Returns:
        quantized_weight: [out_channels, in_channels]
    """
    # 원본 dtype 보존
    dtype = weight.dtype
    
    # Quantization 범위 계산
    # 예: 8bit -> -128 ~ 127
    q_max = 2 ** (n_bit - 1) - 1
    q_min = -(2 ** (n_bit - 1))
    
    # 1. Channel별 Max 값 계산 (Vectorized)
    # dim=1 (row) 방향으로 max를 구하고 차원 유지 [out_channels, 1]
    # abs() 후 max를 구합니다.
    w_max = weight.abs().amax(dim=1, keepdim=True)
    
    # 2. Scale 계산
    # w_max가 0인 경우(모든 가중치가 0) 나눗셈 에러 방지를 위해 1.0으로 대체
    scale = w_max / q_max
    scale = torch.where(scale == 0, torch.tensor(1.0, dtype=dtype, device=weight.device), scale)
    
    # 3. Quantize (Broadcasting 적용)
    # [out, in] / [out, 1] 연산이 자동으로 row별로 적용됨
    weight_scaled = weight / scale
    
    q_weight = torch.clamp(
        torch.round(weight_scaled), 
        q_min, 
        q_max
    )
    
    # 4. Dequantize
    quantized = q_weight * scale
    
    return quantized.to(dtype)

def quantize_weight_grouped(weight: torch.Tensor, n_bit: int, group_size: int = 128) -> torch.Tensor:
    """
    Group 단위로 scale을 계산하여 quantization (Vectorized Version)
    
    Args:
        weight: [out_channels, in_channels]
        n_bit: 양자화 bit 수
        group_size: 그룹 크기 (기본값 128)
    
    Returns:
        quantized_weight: [out_channels, in_channels] (원본 shape 복구됨)
    """
    dtype = weight.dtype
    out_channels, in_channels = weight.shape
    
    # Group Size 유효성 검사
    if in_channels % group_size != 0:
        raise ValueError(f"Weight in_channels ({in_channels}) must be divisible by group_size ({group_size}).")
    
    # 1. Grouping을 위한 View (Reshape)
    # [Out, In] -> [Out, Num_Groups, Group_Size]
    # 메모리 복사 없이 stride만 변경되므로 매우 빠름
    weight_grouped = weight.view(out_channels, -1, group_size)
    
    # Quantization 범위 계산
    q_max = 2 ** (n_bit - 1) - 1
    q_min = -(2 ** (n_bit - 1))
    
    # 2. Group별 Max 값 계산
    # 마지막 차원(group_size)을 기준으로 Max를 구함 -> [Out, Num_Groups, 1]
    w_max = weight_grouped.abs().amax(dim=-1, keepdim=True)
    
    # 3. Scale 계산
    # w_max가 0인 경우 방어 로직
    scale = w_max / q_max
    scale = torch.where(scale == 0, torch.tensor(1.0, dtype=dtype, device=weight.device), scale)
    
    # 4. Quantize (Broadcasting)
    # [Out, Groups, 128] / [Out, Groups, 1] 연산이 자동으로 수행됨
    weight_scaled = weight_grouped / scale
    
    q_weight = torch.clamp(
        torch.round(weight_scaled), 
        q_min, 
        q_max
    )
    
    # 5. Dequantize & Restore Shape
    quantized_grouped = q_weight * scale
    
    # 다시 원래 [Out, In] 형태로 복구
    quantized = quantized_grouped.view(out_channels, in_channels)
    
    return quantized.to(dtype)

def quantize_weight_optimized(weight: torch.Tensor, n_bit: int, grid_search: bool = True) -> torch.Tensor:
    """
    MSE Error를 최소화하는 최적의 Scale을 찾는 Quantization (Vectorized)
    """
    dtype = weight.dtype
    device = weight.device
    out_channels = weight.shape[0]
    search_steps = 10

    # Quantization 파라미터
    q_max = 2 ** (n_bit - 1) - 1
    q_min = -(2 ** (n_bit - 1))
    
    # 1. 기준이 되는 Max 값 계산 [out, 1]
    # 절대값의 최대값 (AbsMax)
    w_abs = weight.abs()
    w_max = w_abs.amax(dim=1, keepdim=True)
    w_max = torch.clamp(w_max, min=1e-5) # 0 나누기 방지

    # 2. 최적의 Scale을 저장할 변수 초기화
    # 초기값은 Max Scaling (ratio=1.0)으로 설정
    best_scale = w_max / q_max
    
    # 초기 에러는 무한대로 설정
    best_err = torch.full((out_channels, 1), float('inf'), device=device, dtype=dtype)
    
    # 3. Grid Search (순차적으로 실행하여 메모리 절약)
    # 1.0부터 0.9까지 search_steps 단계로 탐색
    # 예: [1.0, 0.99, 0.98, ... 0.90]
    search_ratios = torch.linspace(1.0, 0.9, search_steps, device=device)
    
    for ratio in search_ratios:
        # (1) 현재 Ratio에 대한 Scale 계산
        current_scale = (w_max * ratio) / q_max

        # (2) Quantize & Dequantize 시뮬레이션
        # 메모리 절약을 위해 임시 변수 사용
        q_weight = torch.clamp(torch.round(weight / current_scale), q_min, q_max)
        dequantized = q_weight * current_scale

        # (3) MSE Error 계산 (Channel별 평균)
        # (Original - Dequantized)^2
        current_err = (weight - dequantized).pow(2).mean(dim=1, keepdim=True)
        
        # (4) 더 좋은(에러가 작은) Scale 발견 시 업데이트
        # 에러가 더 작은 채널의 인덱스 찾기
        better_mask = current_err < best_err
        
        # 해당 인덱스의 scale과 error 업데이트
        best_scale = torch.where(better_mask, current_scale, best_scale)
        best_err = torch.where(better_mask, current_err, best_err)

        # (선택) 메모리 즉시 해제 (필수는 아니지만 도움됨)
        del q_weight, dequantized, current_err, better_mask

    # 4. 최종적으로 찾은 best_scale로 Quantization 수행
    final_q = torch.clamp(torch.round(weight / best_scale), q_min, q_max)
    quantized = final_q * best_scale

    return quantized.to(dtype)

class QuantizationManager:
    """Layer별 bit 선택 + Channel별 quantization 관리자"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.base_model = None
        self.tokenizer = None
        self.original_weights = {}
    
    def load_base_model(self):
        """원본 모델 1회만 로드"""
        if self.base_model is None:
            print("  [Quant] Loading base model (one-time)...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path
            )
            
            # 원본 weight 백업
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear):
                    self.original_weights[name] = module.weight.data.clone()
            
            print(f"  [Quant] Backed up {len(self.original_weights)} layer weights")
    
    def apply_config(self, bit_config: Dict[str, int]):
        """
        Layer별 bit_config 적용 (channel-wise quantization)
        
        Args:
            bit_config: {layer_name: bit} - layer별 단일 bit 값
                       해당 layer의 모든 channel을 이 bit로 quantize
        """
        self.load_base_model()
        
        quantized_count = 0
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and name in bit_config:
                if name in self.original_weights:
                    # 원본으로 복원
                    module.weight.data = self.original_weights[name].clone()
                    
                    # Per-channel quantization 적용
                    n_bit = bit_config[name]
                    with torch.no_grad():
                        quantized = quantize_weight_grouped(module.weight.data, n_bit)
                        module.weight.data = quantized
                    quantized_count += 1
        
        print(f"  [Quant] Applied to {quantized_count}/{len(bit_config)} layers (per-channel scale)")

        return self.base_model
