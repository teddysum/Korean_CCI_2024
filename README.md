# 대화 맥락 추론 Baseline
본 소스 코드는 '국립국어원 인공지능의 한국어 능력 평가' 시범 운영 과제 중 '대화 맥락 추론'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드입니다.  

학습, 추론 및 평가는 아래의 실행 방법(How to Run)에서 확인하실 수 있습니다.  

|Model|Accuracy|
|:---:|---:|
|MLP-KTLim/llama-3-Korean-Bllossom-8B|0.0|

## 리포지토리 구조 (Repository Structure)
```
# 학습에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
├── test.py
└── train.py

# 학습에 사용될 커스텀 함수들을 보관하는 디렉토리
src
├── data.py     # Custom Dataset
└── utils.py
```

## 데이터 형태 (Data Format)
```
{
    "id": "nikluge-2024-대화 맥락 추론-train-000001",
    "input": {
        "conversation": [
            {
                "speaker": 2,
                "utterance": "진짜 신의 한수",
                "utterance_id": "MDRW2100003410.1.1"
            },
            {
                "speaker": 1,
                "utterance": "이사하자마자 비 많이 와서 베란다 물 많이 새는 거 알았잖아",
                "utterance_id": "MDRW2100003410.1.2"
            },
            {
                "speaker": 2,
                "utterance": "글치 계속 해떴으면 몰랐겠지",
                "utterance_id": "MDRW2100003410.1.3"
            },
            ...
            ...
            ...
        ],
        "reference_id": [
            "MDRW2100003410.1.11"
        ],
        "category": "원인",
        "inference_1": "화자2가 사는 곳 근처에서 베란다 보수 공사가 진행되고 있다.",
        "inference_2": "화자2가 사는 곳 근처에서 싱크홀 보수 공사가 진행되고 있다.",
        "inference_3": "화자2가 사는 곳 근처에서 싱크홀 보수 공사가 중단되었다."
    },
    "output": 2     # The Correct answer is inference_2
}
```
## 실행 방법 (How to Run)
### 학습 (Train)
```
python -m run.train \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0
```
[학습에 대한 설명]

### 추론 (Inference)
```
python -m run.test \
    --output output.txt \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0
```
[추론에 대한 설명]


## Reference

huggingface/transformers (https://github.com/huggingface/transformers)  
Bllossome (Teddysum) ((https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
