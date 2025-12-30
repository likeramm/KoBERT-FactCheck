# 🕵️‍♂️ KoBERT-FactCheck

> **KoBERT 기반 한국어 요약문 사실성 검증 및 환각 탐지 모델** > *Korean Fact Verification Model using KoBERT Fine-tuning*

## 📝 Project Overview
이 프로젝트는 **KorNLI 데이터셋**을 학습한 KoBERT 모델을 사용하여, 원문(Premise)과 요약문(Hypothesis) 사이의 논리적 관계(함의, 모순, 중립)를 판단합니다. 이를 통해 생성형 AI가 만든 요약문의 **거짓 정보(Hallucination)를 탐지**하는 것을 목표로 합니다.

## 🛠 Tech Stack
- **Model:** SKT KoBERT (Fine-tuned)
- **Data:** KorNLI (SNLI, MultiNLI)
- **Library:** PyTorch, Hugging Face Transformers, SentencePiece
- **Task:** Natural Language Inference (NLI)