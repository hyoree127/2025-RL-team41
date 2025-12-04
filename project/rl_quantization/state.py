# state.py

import torch
from typing import Dict
from transformers import AutoModelForCausalLM

class ActivationCollector:
    """Layer별 activation 통계 수집"""
    
    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self.stats = {}
        self.hooks = []
    
    def register_hooks(self):
        """Linear layer에 forward hook 등록"""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._collect(n, inp[0])
                )
                self.hooks.append(hook)
    
    def _collect(self, name: str, activation: torch.Tensor):
        """Activation 통계 수집 (layer 전체 평균)"""
        # activation shape: [batch, seq_len, hidden_dim]
        act = activation.detach().float()
        
        # Layer 전체 평균 계산 (모든 차원에 대해)
        layer_mean = act.abs().mean().item()  # scalar 값
        
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(layer_mean)
    
    def get_state(self) -> Dict[str, float]:
        """최종 통계 반환 (layer별 scalar)"""
        state = {}
        for name, values in self.stats.items():
            state[name] = sum(values) / len(values)  # 평균
        return state
    
    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.stats.clear()