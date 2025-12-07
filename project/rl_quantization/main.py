# main.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import glob
import gc
from config import Config
from state import ActivationCollector
from policy import QuantizationPolicy
from reward import RewardCalculator
from trainer import RLTrainer
from quantizer import QuantizationManager

def load_data(SEED):
    """Calibration 및 Evaluation 데이터 로드"""
    data_files = glob.glob(Config.DATASET_PATH)
    dataset = load_dataset("json", data_files=data_files, split="train").shuffle(seed=SEED)
    
    calib_data = dataset.select(range(Config.CALIB_SAMPLES))
    eval_data = dataset.select(range(Config.CALIB_SAMPLES, Config.CALIB_SAMPLES + Config.EVAL_SAMPLES))
    
    return calib_data, eval_data

def collect_activation_stats(model, tokenizer, calib_data):
    """Activation 통계 수집 (layer별)"""
    collector = ActivationCollector(model)
    collector.register_hooks()
    
    model.eval()
    print("  Collecting activation statistics...")
    with torch.no_grad():
        for idx, sample in enumerate(calib_data):
            inputs = tokenizer(
                sample['text'], 
                return_tensors='pt', 
                truncation=True, 
                max_length=Config.MAX_SEQ_LENGTH
            ).to(Config.DEVICE)
            model(**inputs)
            
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(calib_data)} samples...")
    
    states = collector.get_state()
    collector.clear()
    
    return states

def print_config_analysis(config_path, title):
    """Config 분석 출력"""
    config = torch.load(config_path)
    from collections import Counter
    
    bit_dist = Counter(config.values())
    avg_bit = sum(config.values()) / len(config)
    
    print(f"\n{title}")
    print(f"  Total layers: {len(config)}")
    print(f"  Average bit: {avg_bit:.2f}")
    print(f"  Bit distribution:")
    print(f"    W3: {bit_dist.get(3, 0):3d} layers ({bit_dist.get(3, 0)/len(config)*100:5.1f}%)")
    print(f"    W4: {bit_dist.get(4, 0):3d} layers ({bit_dist.get(4, 0)/len(config)*100:5.1f}%)")
    print(f"    W8: {bit_dist.get(8, 0):3d} layers ({bit_dist.get(8, 0)/len(config)*100:5.1f}%)")
    print(f"  Config file: {config_path}")

def main(SEED):
    print("="*50)
    print("RL-based Layer-wise Mixed-Precision Quantization")
    print("="*50)

    np.random.seed(SEED)
    
    # 1. 데이터 로드
    print("\n[1/7] Loading data...")
    calib_data, eval_data = load_data(SEED)
    print(f"  Calibration samples: {len(calib_data)}")
    print(f"  Evaluation samples: {len(eval_data)}")
    
    # 2. 모델 로드
    print("\n[2/7] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map=Config.DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded on {Config.DEVICE}")
    
    # 3. Activation 통계 수집
    print("\n[3/7] Collecting activation statistics...")
    states = collect_activation_stats(model, tokenizer, calib_data)
    print(f"✓ Collected stats for {len(states)} layers")
    
    # 통계 요약 출력
    state_values = list(states.values())
    print(f"  Activation range: [{min(state_values):.4f}, {max(state_values):.4f}]")
    print(f"  Activation mean: {sum(state_values)/len(state_values):.4f}")
    
    # 4. Quantization Manager 초기화
    print("\n[4/7] Initializing Quantization Manager...")
    quant_manager = QuantizationManager(Config.MODEL_PATH)
    
    # 5. RL 컴포넌트 초기화
    print("\n[5/7] Initializing RL components...")
    policy = QuantizationPolicy(num_bits=Config.AVAILABLE_BITS).to(Config.DEVICE)
    reward_calc = RewardCalculator(tokenizer, eval_data)
    trainer = RLTrainer(policy, reward_calc)
    print(f"  Policy network: {sum(p.numel() for p in policy.parameters())} parameters")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    
    # 6. Baseline PPL 측정
    print("\n[6/7] Measuring baseline PPL...")
    baseline_ppl = reward_calc.calculate_ppl(model)
    reward_calc.set_baseline(baseline_ppl)
    print(f"✓ Baseline PPL: {baseline_ppl:.2f}")
    
    # 7. RL 학습
    print("\n[7/7] Starting RL training...")
    print(f"  Episodes: {Config.NUM_EPISODES}")
    print(f"  Alpha (PPL weight): {Config.ALPHA}")
    print(f"  Beta (Memory weight): {Config.BETA}")
    print("-" * 50)

    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    for episode in range(Config.NUM_EPISODES):
        print(f"\n--- Episode {episode+1}/{Config.NUM_EPISODES} ---")
        
        metrics = trainer.train_step(
            states, 
            lambda cfg: quant_manager.apply_config(cfg)
        )
        
        # 최고 성능 업데이트
        updated = trainer.update_best_metrics(
            episode + 1, 
            metrics['reward'], 
            metrics['ppl'],
            metrics['bit_config']
        )
        
        if updated:
            status = []
            if metrics['reward'] == trainer.best_reward:
                status.append("🏆 Best Reward")
            if metrics['ppl'] == trainer.best_ppl:
                status.append("⭐ Best PPL")
            print(f"  {' & '.join(status)}!")
        
        print(f"Episode {episode+1}/{Config.NUM_EPISODES} | "
              f"Reward: {metrics['reward']:.4f} | "
              f"PPL: {metrics['ppl']:.2f} | "
              f"Memory: {metrics['memory_saving']:.2%}")
        
        # 현재까지 최고 기록 표시
        print(f"  [Best Reward] {trainer.best_reward:.4f} (Ep {trainer.best_reward_episode})")
        print(f"  [Best PPL]    {trainer.best_ppl:.2f} (Ep {trainer.best_ppl_episode})")
        
        # 매 10 episode마다 요약
        if (episode + 1) % 10 == 0:
            recent_rewards = trainer.episode_rewards[-10:]
            recent_ppls = trainer.episode_ppls[-10:]
            print(f"\n  === Last 10 Episodes Summary ===")
            print(f"  Avg Reward: {sum(recent_rewards)/len(recent_rewards):.4f}")
            print(f"  Avg PPL: {sum(recent_ppls)/len(recent_ppls):.2f}")
            print(f"  Best Reward: {trainer.best_reward:.4f} (Episode {trainer.best_reward_episode})")
            print(f"  Best PPL: {trainer.best_ppl:.2f} (Episode {trainer.best_ppl_episode})")
    
    # 최종 결과
    print("\n" + "="*50)
    print("Training Completed!")
    print("="*50)
    
    # Best Reward 상세 정보
    print(f"\n{'='*50}")
    print("🏆 BEST REWARD CONFIG")
    print(f"{'='*50}")
    print(f"  Episode: {trainer.best_reward_episode}")
    print(f"  Reward: {trainer.best_reward:.4f}")
    
    # Best Reward Config의 PPL과 Memory 계산
    best_reward_config = torch.load('best_reward_config.pt')
    best_reward_avg_bit = sum(best_reward_config.values()) / len(best_reward_config)
    best_reward_memory_saving = (len(best_reward_config) * 16 - sum(best_reward_config.values())) / (len(best_reward_config) * 16)
    
    # Best Reward episode의 PPL 찾기
    best_reward_ppl = trainer.episode_ppls[trainer.best_reward_episode - 1]
    
    print(f"  PPL: {best_reward_ppl:.2f}")
    print(f"  Degradation: {(best_reward_ppl - baseline_ppl) / baseline_ppl * 100:.2f}%")
    print(f"  Memory Saving: {best_reward_memory_saving*100:.2f}%")
    print(f"  Average bit: {best_reward_avg_bit:.2f}")
    
    print_config_analysis('best_reward_config.pt', '  [Bit Distribution]')
    
    # Best PPL 상세 정보
    print(f"\n{'='*50}")
    print("⭐ BEST PPL CONFIG")
    print(f"{'='*50}")
    print(f"  Episode: {trainer.best_ppl_episode}")
    print(f"  PPL: {trainer.best_ppl:.2f}")
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    print(f"  Degradation: {(trainer.best_ppl - baseline_ppl) / baseline_ppl * 100:.2f}%")
    
    # Best PPL Config의 Reward와 Memory 계산
    best_ppl_config = torch.load('best_ppl_config.pt')
    best_ppl_avg_bit = sum(best_ppl_config.values()) / len(best_ppl_config)
    best_ppl_memory_saving = (len(best_ppl_config) * 16 - sum(best_ppl_config.values())) / (len(best_ppl_config) * 16)
    
    # Best PPL episode의 reward 찾기
    best_ppl_reward = trainer.episode_rewards[trainer.best_ppl_episode - 1]
    
    print(f"  Reward: {best_ppl_reward:.4f}")
    print(f"  Memory Saving: {best_ppl_memory_saving*100:.2f}%")
    print(f"  Average bit: {best_ppl_avg_bit:.2f}")
    
    print_config_analysis('best_ppl_config.pt', '  [Bit Distribution]')
    
    # 비교 표
    print(f"\n{'='*50}")
    print("📊 COMPARISON")
    print(f"{'='*50}")
    print(f"{'Metric':<20} {'Best Reward':<15} {'Best PPL':<15}")
    print(f"{'-'*50}")
    print(f"{'Episode':<20} {trainer.best_reward_episode:<15} {trainer.best_ppl_episode:<15}")
    print(f"{'Reward':<20} {trainer.best_reward:<15.4f} {best_ppl_reward:<15.4f}")
    print(f"{'PPL':<20} {best_reward_ppl:<15.2f} {trainer.best_ppl:<15.2f}")
    print(f"{'Degradation (%)':<20} {(best_reward_ppl - baseline_ppl) / baseline_ppl * 100:<15.2f} {(trainer.best_ppl - baseline_ppl) / baseline_ppl * 100:<15.2f}")
    print(f"{'Memory Saving (%)':<20} {best_reward_memory_saving*100:<15.2f} {best_ppl_memory_saving*100:<15.2f}")
    print(f"{'Average bit':<20} {best_reward_avg_bit:<15.2f} {best_ppl_avg_bit:<15.2f}")
    
    # 모델 저장
    print(f"\n{'='*50}")
    print("💾 SAVING QUANTIZED MODELS")
    print(f"{'='*50}")
    
    # 1. Best PPL Model 저장
    print("\n[1/2] Saving Best PPL Model...")
    best_ppl_model = quant_manager.apply_config(best_ppl_config)
    
    ############################
    ##모델 save best ppl, reward#
    ############################
    
    best_ppl_path = Config.QUANT_OUTPUT_PATH + '_best_ppl'
    best_ppl_model.save_pretrained(best_ppl_path)
    tokenizer.save_pretrained(best_ppl_path)
    print(f"  ✓ Best PPL model saved to: {best_ppl_path}")
    print(f"    - PPL: {trainer.best_ppl:.2f}")
    print(f"    - Degradation: {(trainer.best_ppl - baseline_ppl) / baseline_ppl * 100:.2f}%")
    print(f"    - Memory Saving: {best_ppl_memory_saving*100:.2f}%")
    
    # 2. Best Reward Model 저장
    print("\n[2/2] Saving Best Reward Model...")
    best_reward_model = quant_manager.apply_config(best_reward_config)
    
    best_reward_path = Config.QUANT_OUTPUT_PATH + '_best_reward'
    best_reward_model.save_pretrained(best_reward_path)
    tokenizer.save_pretrained(best_reward_path)
    print(f"  ✓ Best Reward model saved to: {best_reward_path}")
    print(f"    - Reward: {trainer.best_reward:.4f}")
    print(f"    - PPL: {best_reward_ppl:.2f}")
    print(f"    - Memory Saving: {best_reward_memory_saving*100:.2f}%")
    
    print(f"\n{'='*50}")
    print("✓ All models saved successfully!")
    print(f"{'='*50}")
    
    # 저장 경로 요약
    print("\n[Saved Files]")
    print(f"  1. Best PPL Model:    {best_ppl_path}")
    print(f"  2. Best Reward Model: {best_reward_path}")
    print(f"  3. Best PPL Config:   best_ppl_config.pt")
    print(f"  4. Best Reward Config: best_reward_config.pt")
    
    print("\n✓ Training finished!")

if __name__ == "__main__":
    
    for SEED in Config.SEED:
        print("=-"*40)
        print("SEED : ", SEED)
        torch.manual_seed(SEED)
        main(SEED)
