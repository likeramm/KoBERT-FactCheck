import os
import pandas as pd
import torch
from transformers import BertTokenizer

# -----------------------------------------------------------------------
# [1] 필수 라이브러리 체크 및 KoBERT 토크나이저 클래스 정의
# -----------------------------------------------------------------------
try:
    import sentencepiece as spm
except ImportError:
    print("오류: 'sentencepiece' 라이브러리가 설치되지 않았습니다.")
    print("터미널에 다음 명령어를 입력하세요: pip3 install sentencepiece")
    exit()

class KoBERTTokenizer(BertTokenizer):
    def __init__(self, vocab_file, spm_model_file, **kwargs):
        # 1. SentencePiece 모델 로드
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_model_file)
        self.vocab_file = vocab_file
        
        # 2. 대소문자 구분 설정 (KoBERT는 False가 기본)
        if "do_lower_case" not in kwargs:
            kwargs["do_lower_case"] = False
            
        super().__init__(vocab_file, **kwargs)

    def _tokenize(self, text):
        """SentencePiece를 이용해 한글을 쪼개는 핵심 함수"""
        return self.sp_model.EncodeAsPieces(text)

# -----------------------------------------------------------------------
# [2] 설정 및 데이터 로드
# -----------------------------------------------------------------------
DATA_DIR = "./data" # 데이터 폴더 경로

print("\n=== [1] 데이터셋 로드 ===")
# 데이터 로드 함수
def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"로딩: {filename}")
        return pd.read_csv(path, sep='\t', quoting=3, encoding='utf-8')
    return None

# 파일 로드
df_train = None
df_snli = load_data("snli_1.0_train.ko.tsv")
df_mnli = load_data("multinli.train.ko.tsv")

if df_snli is not None and df_mnli is not None:
    df_train = pd.concat([df_snli, df_mnli], ignore_index=True)
    print(f"학습 데이터 병합 완료: {len(df_train)}개")

df_dev = load_data("xnli.dev.ko.tsv")

# -----------------------------------------------------------------------
# [3] 토크나이저 로드 (로컬 파일 사용)
# -----------------------------------------------------------------------
print("\n=== [2] KoBERT 토크나이저 로드 ===")
vocab_path = os.path.join(DATA_DIR, "vocab.txt")
spmodel_path = os.path.join(DATA_DIR, "spiece.model")

if os.path.exists(vocab_path) and os.path.exists(spmodel_path):
    print("로컬 파일(vocab.txt, spiece.model)을 찾았습니다.")
    
    # 위에서 정의한 클래스 사용
    tokenizer = KoBERTTokenizer(
        vocab_file=vocab_path, 
        spm_model_file=spmodel_path
    )
    print("토크나이저 로드 성공!")
else:
    print("오류: data 폴더에 'vocab.txt' 또는 'spiece.model'이 없습니다.")
    exit()

# -----------------------------------------------------------------------
# [4] 토큰화 테스트
# -----------------------------------------------------------------------
print("\n=== [3] 최종 토큰화 테스트 ===")
premise = "말을 탄 사람이 고장난 비행기 위로 뛰어오른다."
hypothesis = "한 사람이 경쟁을 위해 말을 훈련시키고 있다."

print(f"전제: {premise}")

# 토크나이징 실행
inputs = tokenizer(
    premise, 
    hypothesis,
    return_tensors='pt',
    max_length=64,
    padding='max_length',
    truncation=True
)

input_ids = inputs['input_ids'][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print("\n[결과 확인]")
# 보기 좋게 토큰 출력 (PAD 제외)
valid_tokens = [t for t in tokens if t != '[PAD]']
print(f"토큰 리스트: {valid_tokens[:15]} ...")
print(f"ID 리스트 : {input_ids[:10]}")

# 성공 판별
if '[UNK]' not in valid_tokens[:5] and len(valid_tokens) > 5:
    print("\n분리되었습니다.")
else:
    print("\n분리되지않았습니다. 토크나이저 설정을 다시 확인하세요.")

# ===========================================================================
# [4] PyTorch 데이터셋 클래스 정의 (AI 학습용)
# ===========================================================================
from torch.utils.data import Dataset

class KoBERTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        
        # 1. 데이터에서 문장과 정답 가져오기
        # 데이터 파일의 컬럼명(sentence1, sentence2)을 정확히 맞춰야 합니다.
        premise = str(row['sentence1'])   # 전제
        hypothesis = str(row['sentence2']) # 가설
        label = row['gold_label']          # 정답 (entailment, contradiction, neutral)

        # 2. 정답(Label)을 숫자로 변환
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        
        # 가끔 데이터에 이상한 라벨이 섞여있을 수 있어 예외처리
        if label not in label_map:
            # 이상한 데이터는 중립(2)으로 처리하거나 건너뜀 (여기선 일단 2로 처리)
            label_id = 2 
        else:
            label_id = label_map[label]

        # 3. 토크나이징 (두 문장을 합쳐서 인코딩)
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',        # PyTorch 텐서로 반환
            max_length=self.max_len,    # 최대 길이 제한
            padding='max_length',       # 짧으면 0으로 채움
            truncation=True             # 길면 자름
        )

        # 4. 모델에 필요한 결과 반환
        return {
            'input_ids': inputs['input_ids'][0],      # 단어의 숫자 ID
            'attention_mask': inputs['attention_mask'][0], # 실제 단어 위치 표시
            'token_type_ids': inputs['token_type_ids'][0], # 문장 구분 (전제=0, 가설=1)
            'labels': torch.tensor(label_id, dtype=torch.long) # 정답 (0, 1, 2)
        }

# ===========================================================================
# [5] 데이터셋 테스트
# ===========================================================================
print("\n=== [4] 데이터셋 변환 테스트 ===")

# 학습 데이터 중 1000개만 샘플로 테스트 (속도 위해)
# 실제 학습 땐 전체 데이터를 씁니다.
sample_df = df_train[:1000] 

train_dataset = KoBERTDataset(sample_df, tokenizer)

print(f"데이터셋 크기: {len(train_dataset)}개")
print("첫 번째 데이터 샘플 확인:")

# 첫 번째 데이터 하나만 꺼내보기
sample = train_dataset[0]

print(f"- Input IDs 크기: {sample['input_ids'].shape}")
print(f"- Labels (정답): {sample['labels']} (0=함의, 1=모순, 2=중립)")
print("- Input IDs 일부:", sample['input_ids'][:20])

# ===========================================================================
# [5] 모델 로드 및 학습 준비
# ===========================================================================
from transformers import BertForSequenceClassification
from torch.optim import AdamW 
from torch.utils.data import DataLoader

print("\n=== [5] AI 모델 로드 및 학습 설정 ===")

# 1. 디바이스 설정 (맥북 M1/M2/M3 칩 가속 사용)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Mac GPU(MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("NVIDIA GPU(CUDA)")
else:
    device = torch.device("cpu")
    print("CPU를 사용합니다.")

# 2. 모델 불러오기
# NLI는 3가지(함의, 모순, 중립) 중 하나를 맞추는 문제이므로 num_labels=3
model = BertForSequenceClassification.from_pretrained(
    'skt/kobert-base-v1', 
    num_labels=3
)
model.to(device) # 모델을 GPU/CPU로 이동

# 3. 데이터 로더 설정 (한 번에 16개씩 공부)
# 테스트용으로 1000개만 사용해봅니다. (실제 학습 땐 df_train 전체 사용)
train_subset = df_train[:1000] 
train_dataset = KoBERTDataset(train_subset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 4. 최적화 도구(Optimizer) 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

# ===========================================================================
# [6] 학습 루프 (Training Loop) - 테스트용
# ===========================================================================
print("\n=== [6] 학습 시작 (테스트) ===")

model.train() # 학습 모드로 전환

for i, batch in enumerate(train_loader):
    # 1. 데이터를 GPU/CPU로 이동
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # 2. 모델에게 문제 풀리기 (Forward)
    outputs = model(
        input_ids, 
        attention_mask=attention_mask, 
        labels=labels
    )
    
    # 3. 오답노트 확인 (Loss 계산)
    loss = outputs.loss

    # 4. 공부 내용 반영 (Backward)
    optimizer.zero_grad() # 이전 기억 지우기
    loss.backward()       # 틀린 문제 복습
    optimizer.step()      # 지능 업데이트

    # 10번마다 로그 출력
    if i % 10 == 0:
        print(f"Step [{i}/{len(train_loader)}] - 손실(Loss): {loss.item():.4f}")

    # 테스트니까 50번(Step)만 돌리고 멈춤
    if i >= 50:
        print("\n테스트 학습 종료")
        break

# ===========================================================================
# [7] 실전 테스트 (Inference)
# ===========================================================================
import torch.nn.functional as F

def predict_nli(premise, hypothesis):
    """
    두 문장을 받아서 함의(True)/모순(False)/중립(Neutral)을 판단하는 함수
    """
    model.eval() # 평가 모드로 전환 (학습 중단)
    
    # 1. 문장 토큰화
    inputs = tokenizer(
        premise, 
        hypothesis, 
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    )
    
    # 2. GPU/CPU로 데이터 이동
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # 3. 예측 (Gradient 계산 끄기)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits # 점수 (Raw Score)
        probs = F.softmax(logits, dim=1) # 확률로 변환 (0~1)
        
    # 4. 결과 해석
    # 0: Entailment(함의/사실), 1: Contradiction(모순/거짓), 2: Neutral(중립/무관)
    pred_idx = torch.argmax(probs).item()
    pred_prob = probs[0][pred_idx].item() * 100
    
    label_map = {0: '참(Entailment)', 1: '거짓(Contradiction)', 2: '판단불가(Neutral)'}
    result = label_map[pred_idx]
    
    print(f"\n원문: {premise}")
    print(f"요약: {hypothesis}")
    print(f"AI 판단: [{result}] (확신도: {pred_prob:.2f}%)")
    print("-" * 50)

# ---------------------------------------------------------------------------
# 직접 테스트 해보기
# ---------------------------------------------------------------------------
print("\n=== [7] 실전 테스트 시작 ===")

# 테스트 케이스 1 (사실인 경우)
predict_nli(
    "축구 경기장에서 선수들이 공을 차고 있다.", 
    "선수들이 축구를 하고 있다."
)

# 테스트 케이스 2 (거짓인 경우)
predict_nli(
    "사람들이 맑은 날씨에 공원에서 산책을 하고 있다.", 
    "지금 밖에는 비가 억수같이 쏟아지고 있다."
)

# 테스트 케이스 3 (무관한 경우)
predict_nli(
    "나는 오늘 점심으로 햄버거를 먹었다.", 
    "미국의 대통령은 백악관에 산다."
)