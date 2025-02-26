# FLAN

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
