<img width="680" alt="image" src="https://github.com/user-attachments/assets/fddbb1c2-0329-4d96-9c1d-1bc5b048ae4c" /># FLAN

"Finetuned Language Models Are Zero-Shot Learners" - 2022

ㅡ Alexander Wei, Maarten Bosma

ㅡ Google Research

[Paper to Read](https://arxiv.org/pdf/2109.01652)



# 1. Introduction
기존 GPT-3(Brown et al., 2020)은 Few-Shot Learning에는 매우 강력하지만, Zero-Shot Learning 즉, 학습하지 않은 Unseen task에 서는 그 성능이 크게 떨어지는 경우가 많았습니다.

이는 프롬프트가 사전 학습된 데이터와 다르게 구성되기 때문이라고 지적합니다.

그래서 FLAN의 저자는 Zero-Shot의 성능을 향상시키기 위해 Instruction Tuning (Instruction을 통해 설명된 Dataset에 대한 Fine-Tuning) 기법을 소개하면서, 모델이 다양한 자연어 지시문을 이해하고 이를 통해서 GPT-3 보다 작은 모델을 사용해도 Unseen task에서 좋은 성능을 보일 수 있음을 소개하고자 합니다.


## 2. FLAN (Finetuned Language Net)
이 논문에서는 대형 언어 모델의 Zero-Shot 성능을 향상시키기 위한 간단한 방법을 제안하며, 137B 파라미터의 모델을 미세 조정한 결과를 FLAN(Finetuned Language Net)이라는 이름으로 소개합니다.

<img src="https://github.com/user-attachments/assets/f23fb10f-c790-4081-9c59-fd4c967b11d3" width=700>

### 2.1.1. Pretrain-finetune
Fine-Tuning은 각 Task별로 많은 Task-specific 예제가 필요하며, Task마다 별도의 특화된 모델을 생성합니다.
이때 Pre-trained Language Model (PLM)의 모든 파라미터를 업데이트하면서 Fine-Tuning 합니다.
그 결과 해당 Task에 대해 매우 높은 성능을 보이지만, 레이블된 데이터가 반드시 필요하다는 단점이 있습니다.

### 2.1.2. Prompting
PLM을 그대로 사용하면서, Few-Shot, One-Shot, Zero-Shot 방식으로 프롬프트를 제사히여 Task 성능을 끌어올리는 방법입니다.
디양한 Task에 적응할 수 있는 장점이 있으나, 일반적으로 Fine-Tuning에 비해서 성능이 다소 낮은 경향이 있습니다.

### 2.1.3. Instruction Tuning (FLAN)
PLM에 다양한 Task에 대한 NLP Instruction을 제공하면서 학습시키는 방법입니다.
이를 통해서 모델은 주어진 Instruction을 이해하고 따르는 방법을 학습하며, Unseen Task에서도 Instruction 만으로도 적덜한 답변을 생성할 수 있습니다.
Fine-Tuning과 Prompting의 장점을 결합해서, 적은 데이터로도 여러 Task에서 높은 성능을 발휘하는 방법론입니다.


## 3. Instruction Tuning
Instruction Tuning의 방법은 Input에 Instruction을 넣어줘서 학습을 시키는 과정입니다. 아래 그림에서 표시한 부분이 Instruction 부분이며, 이 Instruction을 어떻게 넣어주는 거에 따라서 답변의 성능이 증가할 수 있습니다.

논문에서는 다양한 방식으로 Instruction을 추가해 학습시키며, 이러한 방법이 모델이 다양한 NLP task를 보다 잘 수행하도록 돕는다고 설명하고 있습니다.

특히, 기존의 Supervised Learning을 활용하여 Instruction을 따르는 학습하도록 하는 것이 핵심입니다. 이를 통해, 모델은 Unseen Task에서도 일반화할 수 있도록 만듭니다.

<img src="https://github.com/user-attachments/assets/dae4b6e7-42d4-4215-b109-1340ea6dbe63" width=700>

### 3.1. Task & Templates

### 3.1.1. Dataset and Task Clustering
Instruction Tuning을 위해서 새로운 데이터셋을 직접 만드는 것은 매우 비효율적이어서, 기존 연구에서 사용한 데이터셋을 변환해서 사용합니다.

아래 그림과 같이, TensorFolw Datasets에서 재공하는 62개의 NLP 데이터셋을 12개의 Task Clusters로 나누고, 파란색은 NLU (Natural Language Understanding) Task로 민트색은 NLG (Natural Langugate Generation) Task으로 나뉩니다.

<img src="https://github.com/user-attachments/assets/f57e71dd-d275-417b-abca-a43376f0eba2" width=700>

#### Dataset 정리

| Task Cluster                                    | Count        | Dataset List                                                                                               |
|-------------------------------------------------|--------------|-------------------------------------------------------------------------------------------------------------|
| NLI                                             | 7개          | CB, ANLI (R1-R3), MNLI, QNLI, RTE, SNLI, WNLI                                                                   |
| Commonsense Reasoning                           | 4개          | HellaSwag, CoPA, PiQA, StoryCloze                                                                              |
| Sentiment Analysis                              | 4개          | Sent140, IMDB, SST-2, Yelp                                                                                     |
| Struct-to-Text                                  | 4개          | DART, CommonGen, E2ENLG, WEBNLG                                                                                |
| Closed-Book QA                                  | 3개          | NQ, ARC (easy/challenge), TQA                                                                                  |
| Coreference Resolution                          | 3개          | Winogrande, DPR, WSC273                                                                                        |
| Translation                                     | 8개          | ParaCrawl (EN/ES, EN/DE, EN/FR), WMT-16 (EN/CS, EN/DE, EN/FI, EN/RO, EN/RU, EN/TR)                                |
| Summarization                                   | 11개         | AG News, AESLC, CNN-DM, Gigaword, Multi-News, Newsroom, Opin-Abs (iDebate, Movie), SamSum, Wiki Lingua EN, XSum  |
| Reading Comprehension                           | 5개          | DROP, BoolQ, MultiRC, OBQA, SQuAD                                                                              |
| Paraphrase/Similarity Matching                  | 4개          | QQP, MRPC, PAWS, STS-B                                                                                         |
| Reading Comprehension with Commonsense          | 2개          | CosmosQA, ReCoRD                                                                                              |
| Miscellaneous                                   | 7개          | QuAC, CoQA, WIC, TREC, CoLA, Math, Fix Punctuation (NLG)                                                       |

### 3.1.2. Instruction Template
각 데이터셋에 대해 Natural Language Instruction을 사용해서 10개의 서로 다른 Template를 수동으로 작성합니다.
대부분의 Template은 원래 Task 그대로 설명하지만, 다양성을 위해 각 데이터셋에 대해 최대 3개의 Template task의 관점을 반대로 제시합니다.
이후 각 데이터셋들은 다양한 표현방식이 섞이게끔 10개의 템플릿 중에서, 학습할 때마다 하나를 무작위로 선택해서 해당 데이터셋의 예제를 구성합니다.

아래 그림은 NLI 데이터셋의 Instruction Template 예시입니다.

<img src="https://github.com/user-attachments/assets/32a8242f-e8b4-4869-b6fa-1d893e313b92" width=700>

#### NLI 데이터셋 (Left)
```text
Premise : 러시아 우주비행사 발레리 폴랴코프는 1994~1995년 동안 438일 동안 우주에서 머문 최장 기록을 세웠다.
Hypothesis : 러시아인은 우주에서 가장 오래 머문 기록을 가지고 있다.
```

#### NLI Instruction Template [1 ~ 3] (Right)
```text
Template 1 : 위 문장을 읽고, 가설이 전제에서 유추될 수 있는지 판단하세요.
Template 2 : 전제(premise)와 가설(hypothesis)이 주어졌을 때, 전제로부터 가설을 도출할 수 있는지 판단하세요.
Template 3 : 다음 질문에 답하세요: 전제에 따라 가설을 추론할 수 있습니까?
```

### 3.1.3. Classification with Options
FLAN은 디코더 기반(Decoder-Only) 언어 모델로, Free Text 생성 방식으로 정해진 형식 없이 사용자가 원하는 문장을 생성할 수 있습니다.
따라서 텍스트 생성 Task에는 별도 수정 없이 사용할 수 있습니다.
기존 GPT-3은 Rank Classification 방식을 사용하여 'Yes' 또는 'No'의 출력 중 더 높은 확률을 가진 것을 모델의 예측값으로 선택하는 방식을 사용합니다.

그러나 'Yes'의 답변을 표현하는 방식이 다양하다면 (ex. 'sure', 'correct' etc.), 'Yes'의 확률이 분산되어 확률값이 낮아질 수 있습니다.
FLAN에서는 Options 토큰을 추가해서 이를 해결합니다.
질문에 대한 선택지를 직접 제공해서, 모델이 원하는 응답이 어떤 것인지 알 수 있도록 하는 방법 입니다.
예를들어, 정답을 양자택일 할 수 있도록 미리 제안하는 방법이라 생각하면 됩니다.

#### NLI Dataset with Options
```text
Premise : 러시아 우주비행사 발레리 폴랴코프는 1994~1995년 동안 438일 동안 우주에서 머문 최장 기록을 세웠다.
Hypothesis : 러시아인은 우주에서 가장 오래 머문 기록을 가지고 있다.
OPTIONS :
- yes
- no
```

### 3.2. Evaluation Splits
기존 연구에서는 모델을 평가할 때, 훈련 데이터에 없는 특정 데이터셋을 배제하는 방식으로 Zero-Shot 학습을 평가했습니다.

그러나, FLAN에서는 더 보수적인 평가 기준을 사용함으로 평가 시점에 특정 Task Cluster를 완전히 제외하여, 해당 클러스터에 속하는 모든 데이터셋델은 NLI를 제외한 모든 태스크로 훈련된 FLAN을 사용하여 NLI 성능을 평가합니다.

ex) 자연어 추론(NLI) 태스크의 제로샷 성능을 평가하려면, NLI 관련 모든 데이터셋(MNLI, QNLI 등)을 학습에서 제외하고 평가

12개의 태스크 클러스터 중 c개 클러스터를 평가하려면, c개의 모델을 따로 훈련해야 합니다.
즉, 각 모델은 특정 클러스터를 제외하고 훈련하며, 이후 제외된 클러스터에서 성능을 평가합니다.

ex) 자연어 추론(NLI) 성능 평가 모델은 NLI를 제외한 모든 태스크로 훈련된 FLAN을 사용하여 NLI 성능을 평가


### 3.3 Experiment
#### 1. Base Model
   
   LaMDA-PT 모델을 Instruction Tuning하여 FLAN 생성

#### 2. 데이터셋 구성 및 샘플링
   
   모든 데이터셋을 섞어서 (randomly sampled) 훈련 데이터로 사용

   각 데이터셋에서 최대 30,000개 예제 사용 (데이터셋 크기 균형 유지)

   데이터셋 혼합 비율은 예제 개수 비례 방식(Examples-Proportional Mixing Scheme) 적용 (최대 3,000개의 샘플 사용)

#### 3. 학습 설정

   총 30,000 스텝(Gradient Steps) 동안 학습 진행

   배치 크기: 8,192 Token

   최적화 기법: Adafactor Optimizer (Shazeer & Stern, 2018)

   학습률: 3e-5 (0.00003)

   입력 시퀀스 길이: 1024 토큰, 타겟 시퀀스 길이: 256 토큰

   Packing 기법 적용: 여러 개의 샘플을 하나의 입력 시퀀스로 묶고, EOS(Token End) 토큰을 사용해 입력과 타겟 구분

#### 4. 학습 시간 및 평가

   TPUv3 (128코어)에서 약 60시간 소요

   모든 평가는 30K 스텝 학습 후 마지막 체크포인트에서 진행


### 3.4 Result
### Task Cluster 별 성능
NLI, Reading Comprehension, Closed-Book QA, Translation, Commonsense Reasoning, Coreference Resolution, Struct-to-Text 데이터셋에 대한 FLAN을 평가합니다.

3.2. Evaluation Splits에 설명된 대로 데이터셋을 Task Cluster로 그룹화 하고, 특정 Task Cluster를 제외하고 학습한 후, 제외된 Task Cluster에서 Zero-Sht 성능을 평가합니다.

ex) NLI 성능을 평가할 때는, FLAN에서는 NLI 데이터셋을 전혀 학습하지 않은 상태로 평가 합니다.

이 방법을 통해서, 각 데이터셋에서 모든 Template의 평균 성능을 측정합니다.
즉, 특정한 프롬프트 구조에 의존하지 않고, 일반적인 자연서 Instruction에 대한 기대 성능을 평가하는 것 입니다.

모든 Task 별로 FLAN 137B 모델이 우수한 성능을 띄고 있다.
<img src="https://github.com/user-attachments/assets/faae9e62-eaf8-49cd-abad-ea3b6181e3df" width=700>

### GPT-3 VS FLAN
GPT-3 175B Zero-Shot, GPT-3 175B Few-Shot, FLAN 137B Zero-Shot에 대하여 3가지 Task에 대해서 성능을 비교했으며, FLAN이 성능이 다음과 그림과 같이 좋았습니다.
<img src="https://github.com/user-attachments/assets/29f0d83b-077c-4e44-bd0e-e9c53e4029b5" width=700>

## 4. Ablation Studies & Further Analysis
### 4.1 Number Of Instruction Tuning Clusters
더 많은 Task Cluster를 사용하여 Instruction Tuning을 수행할수록 Zero-Shot 성능이 향상되었습니다.

아래 그림은 3개의 Task Clustering(NLI, Colsed-Book QA, Commonsense Reasoning)을 제외하고 평가를 실험햇습니다. 나머지 7개의 Cluster를 Instruction Tuning에 사용하였고, Instuction Tuning에 사용한 Clusterdml 수를 1개에서 7개 까지 점진적으로 증가시키면서 성능 변화를 평가했습니다.

<img src="https://github.com/user-attachments/assets/350b5eb2-0125-415c-99da-fe3bc7ee7f8c" width=500>

Sentimetn Analysis Cluster에는 성능 향상이 거의 없었다.

### 4.2 Scaling Laws
Instruction Tuning은 모델 크기가 클수록 Zero-Shot 성능이 크게 향상하지만, 작은 모델인 경우에는 오히려 성능이 줄어들었습니다.

다음은 동일한 Instruction Tuning Clusterfmf 사용하고 모델크기를 (422M, 2B, 8B, 68B, 137B) 점진적으로 증가킴으로 Zero-Shot을 비교했습니다.

<img src="https://github.com/user-attachments/assets/f49d37d7-f8ad-4f4c-8a73-2beb727cf5f3" width=500>

8B 이하의 작은 모델에서는 Instriction Tuning 데이터셋(40개 Task) 학습만으로 그 모델의 용량을 넘어 버리므로, 새로운 Task에 일반화 하기 어려워 그 성능이 줄어드는 문제가 발생합니다.

### 4.3 Role of Instructions
Instruction Tuning이 Zero-Shot 성능을 향상시키는 핵심인지 실험적으로 증명합니다.

다음 그림은 4 가지의 실험을 통해 평가를 비교합니다.

<img src="https://github.com/user-attachments/assets/10e09522-9d6a-4892-bdc1-f66298b6e61f" width=500>

Fintuning datset에 Instruction Tuning이 적용한것과 Eval dataset에 Instruction Tuning이 적용된 FLAN이 가장 성능이 좋았습니다.

### 4.4 Instruction with Few-Shot Exampels
여기까지 FLAN 모델의 Zero-Shot 성능에 대해서 알아봤습니다. 4.4에서는 Few-Shot의 성능과 비교해서, 더 많은 정보가 주어진 Few-Shot에서는 Zero-Shot 보다 더 좋은 성능을 보였습니다.

<img src="https://github.com/user-attachments/assets/f532a427-5cce-46b5-ae4f-f6460f9869dd" width=700>
