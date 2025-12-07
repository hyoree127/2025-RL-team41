# trainer.py
import torch
import torch.optim as optim
from typing import Dict
from config import Config
from policy import QuantizationPolicy
from reward import RewardCalculator
from transformers import AutoModelForCausalLM

class RLTrainer:
    """REINFORCE 알고리즘 기반 학습"""
    
    def __init__(self, policy: QuantizationPolicy, reward_calculator: RewardCalculator):
        self.policy = policy
        self.reward_calculator = reward_calculator
        self.optimizer = optim.Adam(policy.parameters(), lr=Config.LEARNING_RATE)
        
        self.episode_rewards = []
        self.episode_ppls = []
        self.reward_history = []
        
        # 최고 성능 추적
        self.best_reward = float('-inf')
        self.best_reward_episode = 0
        self.best_ppl = float('inf')
        self.best_ppl_episode = 0
    
    def train_step(self, states: Dict[str, float], quantize_fn):
        """1회 에피소드 학습"""
        self.policy.train()
        
        layer_actions = {}
        layer_log_probs = []
        bit_config = {}
        
        # 각 layer별로 bit 선택
        for layer_name, state_value in states.items():
            state_tensor = torch.tensor(state_value, dtype=torch.float32).to(Config.DEVICE)
            action, log_prob = self.policy.select_action(state_tensor)
            
            layer_actions[layer_name] = action
            layer_log_probs.append(log_prob)
            bit_config[layer_name] = self.policy.get_bit(action)
        
        # Quantization 수행
        quant_model = quantize_fn(bit_config)
                
        # Reward 계산
        reward, ppl, memory_saving = self.reward_calculator.get_reward(quant_model, bit_config)
        
        # Policy gradient update
        all_log_probs = torch.stack(layer_log_probs)
        
        # Standardized advantage
        self.reward_history.append(reward)
        
        if len(self.reward_history) > 1:
            reward_mean = sum(self.reward_history) / len(self.reward_history)
            reward_std = (sum((r - reward_mean)**2 for r in self.reward_history) / len(self.reward_history)) ** 0.5
            reward_std = max(reward_std, 1e-8)
            advantage = (reward - reward_mean) / reward_std
        else:
            advantage = 0.0
        
        # Policy loss
        policy_loss = -(all_log_probs * advantage).mean()
        
        # Entropy bonus
        entropy = -(torch.exp(all_log_probs) * all_log_probs).mean()
        entropy_coef = max(0.001, 0.05 * (0.95 ** (len(self.episode_rewards) // 10)))
        
        loss = policy_loss - entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        self.episode_rewards.append(reward)
        self.episode_ppls.append(ppl)
        
        print(f"  [Train] Std.Advantage: {advantage:.4f}, Entropy: {entropy:.4f}, Loss: {loss.item():.4f}")

        return {
            'reward': reward,
            'ppl': ppl,
            'memory_saving': memory_saving,
            'loss': loss.item(),
            'advantage': advantage,
            'entropy': entropy.item(),
            'bit_config': bit_config
        }
    
    def update_best_metrics(self, episode: int, reward: float, ppl: float, bit_config: Dict[str, int]):
        """최고 성능 업데이트"""
        updated = False
        
        # 최고 reward 업데이트
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_reward_episode = episode
            torch.save(bit_config, 'best_reward_config.pt')
            updated = True
        
        # 최저 PPL 업데이트
        if ppl < self.best_ppl:
            self.best_ppl = ppl
            self.best_ppl_episode = episode
            torch.save(bit_config, 'best_ppl_config.pt')
            updated = True
        
        return updated
    
    def get_best_config(self, states: Dict[str, float]) -> Dict[str, int]:
        """학습된 policy로 최적 config 추출"""
        self.policy.eval()
        
        bit_config = {}
        with torch.no_grad():
            for layer_name, state_value in states.items():
                state_tensor = torch.tensor(state_value, dtype=torch.float32).to(Config.DEVICE)
                action, _ = self.policy.select_action(state_tensor, deterministic=True)
                bit_config[layer_name] = self.policy.get_bit(action)
        
        return bit_config
