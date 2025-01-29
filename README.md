# USDM
Paralinguistics-Aware Speech-Empowered Large Language Models for Natural Conversation -2024

-Heeseung Kim, Soonshin Seo

[Read the Paper](https://arxiv.org/pdf/2402.05706)

---

# 1. Introduction

## 1.1. Spoken Language Model (SLM)의 한계

대규모 언어 모델(LLM)의 발전으로 텍스트 기반 대화 모델은 크게 발전했습니다.

기존 음성 대화 모델(Spoken Language Model)은 계단식(Cascaded)으로 작동했습니다.

ASR → LLM → TTS (Cascaded 방식)

- ASR(자동 음성 인식) → 입력 음성을 텍스트로 변환
- LLM(텍스트 기반 대화 모델) → 변한된 텍스트를 받아 응답 생성
- TTS(음성 합성) → 생성된 텍스트를 음성으로 변환

## 1.2. SLM의 문제점

### 1. Error Propagation

ASR에서 발생한 오류가 그대로 전파되어 최종 출력이 원본가 달라질 가능성이 높음

### 2. Paralinguistic Information 손실

감정, 억양, 강세 등의 정보가 TTS 과정에서 반영되지 못함

### 3. 텍스트 기반 중간 단계에서 정보 손실

음성을 텍스트로 변환하는 과정에서, Pitch 및 Rhythm 등의 정보가 사라짐

## 1.3. USDM(Unified Spoken Dialog Model)의 목표

논문의 저자들은 USDM을 제안하면서 기존 방식의 한계를 극복하기 위해 다음과 같이 방법론을 제공합니다.

1. ASR-TTS를 모듈 분리 없이, 음성과 텍스트를 하나의 모델에서 학습하는 End-to-End 접근법 도입합니다.

2. Unified Speech-Text Pretraining을 통해 자연스러운 음성-텍스트 변환을 학습합니다.
   
3. Acoustic Units(음향 단위) 기반의 Discrete Speech Representation (이산 음성 표현) 적용으로 감정 및 억양 정보를 유지합니다.

# 2. Related Work

## 2.1. Discrete Speech Representation(이산 음성 표현)

기존 연속 신호 기반 모델들과 달리, 본 논문에서는 이산 음성 표현을 활용해서 음성을 토큰화 했습니다.

### **Discrete Tokens (이산토큰) 이란?**

이산 토큰은 continuous speech signal(연속적인 음성 신호)를 특정 개수의 고정된 심볼로 변환한 것을 의미합니다.

즉, 연속적인 음향 정보를 정해진 개수의 토큰으로 양자화 하여 표현하는 방식

### **이산 토큰을 사용하는 이유**

일반적인 음성 데이터는 **연속적인 신호(Continuous Signal)**입니다. 하지만 자연어에서 사용하는 Tansformer는 Discrete Tokens을 입력으로 받습니다.

따라서 음성을 NLP 모델에서 다루려면 이산 토큰으로 변환이 필요합니다. 결과적으로, 음성-텍스트 변환(ASR), 음성 합성(TTS), 음성 기반 대화 모델(Spoken Dialogue Model) 등에 활용이 가능해집니다.

이산 토큰을 만드는 과정은 두가지 방법이 있습니다.

---

### 1. Self-Supervised Speech Tokens

Self-Supervised Learning 된 음성 표현 모델을 활용해서 연속적인 음성을 이산 토큰으로 변환합니다.

1. 음성을 사전학습된 SSL로 변환하여 연속전인 음성을 표현을 추출합니다.
2. 연속적인 음성표현을 K-means Clustering(k=10,000) 해서 이산적인 클러스터로 변환합니다.
3. 각 클러스터에 ID를 부여해서 정수 형태의 이산적인 토큰을 생성합니다.
- K-means 군집화로 주요 의미 정보를 유지하면서, 압축된 형태로 저장이 가능해집니다.
- Ex) S1, S2, S3, … , S10000 과 같이 정해진 개수의 토큰으로 변환
    
    이 토큰들은 단어가 아닌, Acoustic Units (음향 단위)에 해당합니다.
    

```css
"The cat on the mat" 라는 문장은 다음과 같이 이산 토큰 시퀀스로 변환
-> [S34, S215, S76, S487, S90, S118, S529, S1024]
```

### 2. Neural Audio Codec

Neural Audio Codec을 활용해서 음성을 압축하고 이산적인 벡터로 변환합니다. Autoencoder 기반 압축 방식으로 감정 및 억양 정보까지 유지가 가능해집니다.

1. Autoencdoer 또는 Vector Quantization 을 사용해서 음성을 압축합니다.
2. 압축된 음성을 고정된 개수의 이산 토큰으로 변환합니다.
3. 각 이산 토큰은 음성의 의미뿐만 아니라 감정, 억양, 음색 등 paralinguistic(부가언어적) 요소까지 포함합니다.

```css
"The cat on the mat" 라는 문장을 Neural Auido Codec으로 이산 토큰 시퀀스로 변환
-> [U102, U879, U45, U290, U701, U618, U956, U432]
```

## 2.2. Self-Supervised Vs Neural Audio Codec

**음성의 의미 정보가 주로 필요하고, 구현의 간단함과 계산 효율성이 중요하다면** 

→ K-means Clustering 기반의 Self-Supervised Speech Tokens 학습 방식이 적합합니다.

**음성의 부가언어적 요소까지 세밀하게 반영하여, 자연스러운 음성 합성이나 감정 표현 등을 강화하고자 한다면**

→ Autoencoder 기반의 Neural Audio Codec 방식을 선택하는 것이 더 좋습니다.

USDM 논문에서는 첫 번째 방법으로 Acoustic token화를 선택하였습니다.

## 2.3. USDM이 Self-Supervised Speech Token 방식을 선택한 이유

USDM은 음성을 직접적으로 토큰화하는 방식을 선택했습니다. 그러나 Neural Audio Codec이 아닌 K-means 기반의 Self-Supervised Speech Token 방식을 사용했습니다.

1. 음성 의미 정보 보존
    
    Neural Audio Codec은 부가적인 감정 정보를 잘 보존하지만, 특정 단어의 의미가 달라질 위험이 있습니다.
    
    반면 K-means 기반의 이산 토큰 방식은 음성 데이터를 의미 단위로 유지할 수 있음
    
2. LLM과의 통합 용이성
    
    이산 토큰이 Transformer 모델에서 다루기 쉬운 형태여서 통합의 용이성이 있습니다.
    
    Neural Audio Codec 기반 벡터는 Trnasformer 구조에서 직접 학습하기 어려움
    
3. 대규모 데이터셋에서 학습 용이
    
    USDM은 87,000 시간의 대규모 데이터를 학습해야 하기 때문에, 효율적인 토큰화가 필요합니다.
    
    Neural Audio Codec 방식은 계산량이 많고, 학습이 오래 걸릴 가능성이 큼
    

# 3. Unified Speech-Text Pretraining

### 3.1.1. 데이터 준비 및 토큰화

87,000 시간 분량의 영어 ASR 데이터를 활용 - Multilingual LibriSpeech, GigaSpeech 등

기존 LLM(Mistral-7B) 어휘에 10,000개의 Acoustic unit 토큰과 2개의 특수 토큰을 추가했습니다.

**특수 토큰**

1. **<|correspond|>** : 텍스트와 직접적으로 대응하는 음성 단위가 뒤따름을 명시 하는 토큰
    
    즉, 현재 모달리티(텍스트 or 음성)에서 다른 모달리티로 전환될 때 사용. 
    
    Ex) 음성→ 텍스트 전환 / 텍스트→ 음성 전환
    
2. **<|continue|> :** 동일한 모달리티(음성 → 음성 / 텍스트 → 텍스트) 에서 다음 토큰이 연속됨을 명시하는 토큰
    
    즉, 현재 모달리티가 유지될 때 사용
    
    Ex) 음성→ 음성 연속 / 텍스트→ 텍스트 연속
    

### 3.1.2. Correspodence Relationship Modeling

각 Segment(구간)에서 선택되지 않은 모달리티(음성이 선택되었다면 텍스트)가 50% 확률로 삽입됩니다.

음성이 먼저 선택되었다면, 해당 구간의 나머지 부분에 50% 확률로 대응하는 텍스트 데이터가 추가됩니다. 

반대로 텍스트가 먼저 선택되었다면, 50% 확률로 음성이 추가됩니다.

Ex) 1.  전체 데이터 샘플을 N개의 구간으로 나눔

“The cat on the mat because it was soft and warm”

→ “The cat”, ”on the mat”, ”because it was soft and warm”

1. 각 구간에서 50% 확률로 음성 또는 텍스트를 선택
    
    “The cat” → 텍스트
    
    “on the mat” → 음성
    
    “because it was soft and warm” → 텍스트
    
2. 이 과정을 통해서 음성과 텍스트가 섞인 시퀀스를 생성

## 3.2. Interleaved 음성-텍스트 시퀀스 생성

이 과정을 거쳐 음성과 텍스트가 혼합된(Interleaved) 시퀀스 ${Ij}$를 만들 수 있습니다.

즉, 음성과 텍스트를 분리된 모달리티가 아닌 하나의 시퀀스로 다루도록 구성하는 것 입니다.

이러한 시퀀스를 사용하면 모델이 단순히 음성만(uni-modal) 처리하는 것이 아니라, 음성과 텍스트 간의 관계를 포괄적으로 학습하는 (Cross-modal Modeling) 방식을 학습할 수 있습니다. 이렇게 생성된 시퀀스들은 Loss Function에서 사용되며 음성과 텍스트를 동시에 이해하는 모델을 학습시키는데 사용됩니다. 결과적으로, 이런 방식이 LLM(Mistral-7B)과 결합하여 Unified Spoken Dialog Model(USDM)의 핵심 학습구조가 됩니다.

## 3.3. Method

### 3.3.1. Alignment(음성-텍스트 정렬)

**Montreal Forced Aligner (MFA)**를 활용하여 음성과 해당 텍스트 간의 단어 수준으로 정렬을 수행합니다.

https://www.kaggle.com/code/davidnguyens12/montreal-forced-aligner

---

- **MFA**
    1. 텍스트와 음성 데이터를 입력 받음
    - “The cat on the mat”
    - 음성파일(.wav)을 함께 사용
    1. 음향 모델(Acoustic Model)과 발음 사전(Pronunciation Dictionary)을 활용해 정렬
        
        음향 모델(Acoustic Model) : 음성의 특징 (스펙트로그램, 주파수 등)을 분석하는 모델
        
        발음 사전(Pronunciation Dictonary) : 단어와 음소 간의 관계 저장(”cat” → [k,æ, t])
        
    2. 단어 또는 음소별 시작/ 끝 시간을 추정
    
    ```graphql
    단어 수준 정렬
    The    0.00s - 0.25s
    Cat    0.26s - 0.50s
    On     0.51s - 0.65s
    The    0.66s - 0.80s
    Mat    0.81s - 1.20s
    
    음소 수준 정렬
    k   0.26s - 0.32s
    æ   0.33s - 0.40s
    t   0.41s - 0.50s
    ```
    
    1. 출력된 정렬 정보를 모델 학습에 활용
        
        모델이 음성에서 특정 단어가 등장하는 위치를 알 수 있도록 함
        
        정렬된 데이터는 **음성을 이산 음향 단위(Acoustic Units)로 변환**하는 데 사용
        
    
    Ex) 1. 음성 데이터와 해당 텍스트를 MFA를 통해 정렬
    
    1. 각 단어가 음성 내에서 어디에 위치하는지 정보 획득
    2. 음성 데이터를 50Hz 단위의 이산 토큰으로 변환

### **3.3.2. Correspondence & Continuation Token 적용**

`<|correspond|>`와 `<|continue|>` 토큰을 추가하여 음성과 텍스트 간의 관계를 학습시킵니다.

`<|correspond|>`와 `<|continue|>`는 단순한 구분자가 아니라, 모델이 **음성-텍스트 관계를 명확하게 학습하도록 유도**하는 역할을 합니다.

일반적인 Transformer는 텍스트 기반으로 학습되기 때문에 음성을 추가하면 적절한 연결 관계를 파악하는 것이 어려울 수 있습니다.

하지만 `<|correspond|>` 토큰을 추가하면 **"이전 토큰이 음성이었고, 다음은 텍스트가 나올 것이다"** 라는 신호를 줄 수 있어서 **Cross-Modal** 학습이 가능해집니다.

마찬가지로 `<|continue|>` 토큰이 있으면, 모델이 "**지금 토큰은 같은 모달리티(음성 or 텍스트)로 연속된 데이터이므로, 흐름을 유지해서 처리해야 한다**"는 정보를 학습할 수 있습니다.

EX) “The cat on the mat because it was soft and warm.”

- **`<|correspond|>` (교차 모달 관계)**
    
    서로 다른 모달리티 (음성↔텍스트) 가 연결될 때 사용
    
    ```graphql
    [음성 토큰] <|correspond|> [텍스트 토큰]
    ```
    
- "on the mat" (음성) → "because" (텍스트) 전환 시 `<|correspond|>` 추가
    
    ```graphql
    	[S102, S879, S45] <|correspond|> [T501, T309]
    ```
    
- **`<|continue|>` (동일 모달 관계)**
    
    동일한 모달리티 내에서 시퀀스가 이어질 때 사용
    
    ```css
    [텍스트 토큰] <|continue|> [텍스트 토큰]
    ```
    
- "because" (텍스트) → "it was soft" (텍스트) 연결 시 `<|continue|>` 추가
    
    ```css
    [T501, T309] <|continue|> [T872, T324, T901]
    ```
    

### 3.3.3. Pair-wise Segmentation

기존 Transformer는 순차적인 데이터(텍스트 기반) 학습에 특화되어 있지만, 음성-텍스트를 동시에 다루려면 특정한 데이터 구조가 필요합니다. 그래서 Pair-wise Segmentation을 통해서 음성과 텍스트 데이터를 작은 구간 (segments)로 나누고, 각 구간에서 랜덤하게 음성과 텍스트를 선택하는 방법을 택해야합니다.

Pair-wise Segmentation을 적용하면 다음과 같은 학습 효과를 얻을 수 있습니다.

1. Cross-Modal Understanding
    
    음성과 텍스트를 혼합하여 모델이 두 모달리티를 자연스럽게 연결할 수 있도록 학습
    
2. Nosie Reduction & Generalization
    
    ASR 오타나 TTS 왜곡을 줄이고, 다양한 문맥에서 음성과 텍스트를 동시에 학습 가능
    
3. Multimodal Flexibility (모달리티의 유연성)
    
    특정 상황에서는 텍스트만, 특정 상황에서는 음성만 사용할 수 있도록 데이터 구조 설계
    
    즉, 모델이 음성과 텍스트틀 따로 혹은 함께 사용하는 방법을 학습할 수 있도록 유도
    

**최종 예제: "The cat on the mat because it was soft and warm"**

1. 랜덤하게 구간 나누기 (Pair-wise Segmentation)
    
    ```css
    Segment 1: "The cat"
    Segment 2: "on the mat"
    Segment 3: "because"
    Segment 4: "it was soft and warm"
    ```
    
2. 각 구간에서 음성과 텍스트를 랜덤 선택
    
    ```css
    Segment 1 (텍스트): "The cat"
    Segment 2 (음성): "on the mat"
    Segment 3 (텍스트): "because"
    Segment 4 (음성): "it was soft and warm"
    
    ```
    
3. Correspondence & Continuation 토큰 추가

```css
["The", "cat"] <|continue|> [U302, U450, U500] <|correspond|> ["because"] <|continue|> [U600, U700, U800, U900]

```

이렇게 토큰을 적용하면 모델이 음성과 텍스트 간의 관계를 명확하게 학습할 수 있습니다.

이러한 방식으로 **USDM**이 계단식(Cascaded) ****방식이 아니라**, End-to-End 모델**로 동작할 수 있도록 설계되었습니다.
