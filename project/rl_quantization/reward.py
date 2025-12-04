# reward.py

import torch
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from config import Config
from collections import Counter

class RewardCalculator:
    """보상 계산: PPL + 메모리 절감"""
    
    def __init__(self, tokenizer: AutoTokenizer, eval_dataset: Dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.baseline_ppl = None
    
    def set_baseline(self, ppl: float):
        """원본 모델의 PPL 저장"""
        self.baseline_ppl = ppl
    
    def calculate_ppl(self, model: AutoModelForCausalLM) -> float:
        """
        실제 quantized model의 PPL 계산
        """
        model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for sample in self.eval_dataset:
                inputs = self.tokenizer(
                    sample['text'], 
                    return_tensors='pt',
                    truncation=True,
                    max_length=Config.MAX_SEQ_LENGTH
                ).to(model.device)
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
                count += 1
                
                if count >= Config.EVAL_SAMPLES:
                    break
        
        avg_loss = total_loss / count
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        
        return ppl
    
    def calculate_memory(self, bit_config: Dict[str, int]) -> float:
        """메모리 사용량 추정 (layer별 bit 합)"""
        return sum(bit_config.values())
    
    def get_reward(self, model: AutoModelForCausalLM, bit_config: Dict[str, int]) -> tuple:
        """
        Reward = -α * (PPL - baseline_PPL) + β * memory_saving
        """
        # 실제 quantized model의 PPL 계산
        ppl = self.calculate_ppl(model)
        memory = self.calculate_memory(bit_config)
        
        # 메모리 절감률 계산
        baseline_memory = len(bit_config) * 16
        memory_saving = (baseline_memory - memory) / baseline_memory
        
        ppl_penalty = ppl - self.baseline_ppl if self.baseline_ppl else ppl
        
        reward = -Config.ALPHA * ppl_penalty + Config.BETA * memory_saving
        
        # 디버깅 정보
        bit_dist = Counter(bit_config.values())
        avg_bit = sum(bit_config.values()) / len(bit_config)
        print(f"  [Debug] Avg bit: {avg_bit:.2f}, Bit dist: {dict(bit_dist)}, "
              f"PPL: {ppl:.2f}, Memory saving: {memory_saving:.2%}")
        print(f"  [Debug] PPL penalty: {ppl_penalty:.2f}, Reward: {reward:.4f}")
        
        return reward, ppl, memory_saving