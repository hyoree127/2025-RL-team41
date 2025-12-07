# setting


# config.py

class Config:
    """RL 기반 Mixed-Precision 양자화 설정"""
    
    # ========== 모델 경로 ==========
    MODEL_PATH = ''
    QUANT_OUTPUT_PATH = ''
    
    # ========== 데이터셋 ==========
    DATASET_PATH = ''
    CALIB_SAMPLES = 512          # 캘리브레이션 샘플 수
    EVAL_SAMPLES = 100           # 평가 샘플 수
    MAX_SEQ_LENGTH = 512
    
    # ========== 양자화 설정 ==========
    AVAILABLE_BITS = [4,8]   # 선택 가능한 비트 수
    
    # ========== RL 설정 ==========
    LEARNING_RATE = 2e-3
    NUM_EPISODES = 300           # 학습 에피소드 수
    GAMMA = 0.99                 # 할인율
    
    # ========== 보상 가중치 ==========
    ALPHA = 0.7                # PPL 페널티 가중치
    BETA = 2.0                 # 메모리 절감 보상 가중치
    PENALTY = 0.1              # W8 Layer 수 패널티 퍼센테이지
    
    # ========== 하드웨어 ==========
    DEVICE = "cuda:1"
    SEED = [2322, 4768, 942, 3570, 10964, 64, 20, 382, 651, 3]
