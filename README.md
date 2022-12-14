# Simple-NER-Implementation
한국어 개체명인식기 (BERT based Named Entity Recognition model for Korean)

## 학습 환경
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers|4.24.0-pink?color=FF33CC)](https://github.com/huggingface/transformers)

## 데이터
학습 및 추론 시 사용한 데이터
- [[Naver-CWNU NER Data set]](https://github.com/naver/nlp-challenge/)

### 데이터 전처리
데이터를 받아오고 training, validation, testing set을 구성합니다. 
> **Note** <br>
> default: 0.9 0.05 0.05 
```
bash get_data.sh
```

## 학습 및 추론
BERT 및 RoBERTa 모델에 대해 NER task 학습 및 성능을 평가합니다.
```
bash run_examples.sh
```

## 결과
### BERT-base
<img src=https://user-images.githubusercontent.com/55969260/206592575-91f3039a-07ad-432a-a956-27640b810073.png>

### RoBERTa-base
<img src=https://user-images.githubusercontent.com/55969260/206592626-70d8c755-6a7c-4838-8b31-3a80a8ea3d59.png>
