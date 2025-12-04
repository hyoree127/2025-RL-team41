# policy.py

import torch
import torch.nn as nn
from typing import Dict, List

class QuantizationPolicy(nn.Module):
    """Layer별 bit 선택 policy network"""
    
    def __init__(self, num_bits: List[int] = [3, 4, 8]):
        super().__init__()
        self.num_bits = num_bits
        self.num_actions = len(num_bits)  # 3개
        
        # 간단한 MLP - layer별 activation 평균을 입력받아 bit 선택
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Policy network forward pass
        
        Args:
            state: [1] - layer의 activation 평균값 (scalar를 tensor로)
        
        Returns:
            logits: [num_actions] - 3개 bit 옵션에 대한 logits
        """
        state = state.unsqueeze(0) if state.dim() == 0 else state  # scalar → [1]
        state = state.unsqueeze(-1)  # [1] → [1, 1]
        logits = self.net(state).squeeze(0)  # [1, num_actions] → [num_actions]
        return logits
    
    def select_action(self, state: torch.Tensor, deterministic=False):
        """
        현재 state에서 action(bit) 선택
        
        Args:
            state: scalar tensor - layer의 activation 평균
            deterministic: True면 argmax, False면 샘플링
        
        Returns:
            action: scalar - 선택된 action 인덱스 (0, 1, 2)
            log_prob: scalar - 선택된 action의 log probability
        """
        logits = self.forward(state)  # [num_actions]
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_probs = torch.log_softmax(logits, dim=-1)
        log_prob = log_probs[action]
        
        return action, log_prob
    
    def get_bit(self, action: torch.Tensor) -> int:
        """Action 인덱스를 실제 bit 값으로 변환"""
        return self.num_bits[action.item()]