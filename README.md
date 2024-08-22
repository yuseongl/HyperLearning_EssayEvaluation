# HyperLearning_EssayEvaluation
project for HyperLearning Essay Evaluation using supervised learning model

## 프로젝트 개요
### 주제
에세이 자동 평가 시스템 개발

### 개발자
이유성

### 데이터 소개
- 원천데이터
    - 에세이 글 약 5만개 이상 (대상 초등 고학년 및 중고생)
- 학습데이터
    - 에세이 텍스트
    - 전문 교사가 평가한 에세이 글 평가 점수가 Target 

### 모델 학습
- bert(KLUE) : 에세이의 문맥을 고려하여 embedding 데이터 생성
- LGBM : embedding 데이터와 feature를 이용한 모델 fine tuning
- Mult View Neural Network : embedding 데이터와 feature embedding을 이용한 모델 fine tuning

### EDA
- level 컬럼에 데이터 약 7배의 불균형이 있음을 확인 -> 대부분 level 2에 분포
- etype 컬럼에도 약 3배 불균형 확인 -> 글쓰기 데이터가 가장 많음
- 대부분의 데이터가 25~33정도에 분포
- feature 간의 상관 관계 분석
- 다중 공선성 분석

### 전처리
- 학년 컬럼은 라벨 인코딩 진행
- 글 형식 컬럼은 원-핫 인코딩 진행
- bert 모델을 사용해 문맥 정보가 포함된 embedding 데이터 추출
- 특이점 분해를 이용한 embedding 데이터 차원 축소

### 모델
1. 텍스트 데이터와 feature의 상호작용을 고려한 모델 구현
2. 텍스트 데이터와 feature의 다중공선성을 고려한 multi view 네트워크 구현
3. 잔차 학습을 이용한 모델의 오차 축소 

### 개발 환경
|IDE|GPU 서버|프로그래밍 언어|
|:-----:|:-----:|:-----:|
|<img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"><br/><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"><br/><img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">|4060ti GPU|<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">|
---- -


### 데이터 경로
```
# 데이터 호출 경로
# data/1.Training/라벨링데이터 - JSON 파일 형식
# 해당 경로 안에 embedding_data.csv 존재
```

```
train dataset
data/train/image/감정 - 감정별 디렉터리명 안에 이미지 존재
data/train/label/ - 감정별 라벨링된 JSON 파일
```
```
test dataset
test_set1000/image/감정 - 감정별 디렉터리명 안에 이미지 존재
test_set1000/label - 감정별 라벨링된 JSON 파일
```

## 시작 가이드
### Installation
```
$ git clone https://github.com/yuseongl/HyperLearning_EssayEvaluation.git
$ cd HyperLearning_EssayEvaluation
$ pip install -r requirements.txt
```
### trian model & testing model file
```
# config/config.json에서 학습시킬 모델의 종류와 하이퍼파라미터 튜닝
# models/model_selection.py에서 실행 가능한 모델 확인
$ python train.py
```