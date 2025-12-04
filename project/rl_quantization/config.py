# setting


# config.py

class Config:
    """RL 기반 Mixed-Precision 양자화 설정"""
    
    # ========== 모델 경로 ==========
    MODEL_PATH = '/home/icsl/woori/models/Llama-3-8B-Instruct'
    QUANT_OUTPUT_PATH = '/home/icsl/woori/models/Llama-3-8B-llmcomp4bit-c4-4-8bit'
    
    # ========== 데이터셋 ==========
    DATASET_PATH = '/home/icsl/woori/data/c4/*.jsonl'
    CALIB_SAMPLES = 512          # 캘리브레이션 샘플 수
    EVAL_SAMPLES = 100           # 평가 샘플 수
    MAX_SEQ_LENGTH = 512
    
    # ========== 양자화 설정 ==========
    AVAILABLE_BITS = [3, 4, 8]   # 선택 가능한 비트 수
    
    # ========== RL 설정 ==========
    LEARNING_RATE = 1e-3
    NUM_EPISODES = 50           # 학습 에피소드 수
    GAMMA = 0.99                 # 할인율
    
    # ========== 보상 가중치 ==========
    ALPHA = 0.1                # PPL 페널티 가중치
    BETA = 2.0                  # 메모리 절감 보상 가중치
    
    # ========== 하드웨어 ==========
    DEVICE = "cuda:1"
    SEED = 42