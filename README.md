# Transformer

"Attention Is All You Need"

-Ashish Vaswani, Noam Shazeer

https://arxiv.org/pdf/1706.03762
 
# 1. Introduction

RNN, LSTM, gated recurrent 신경망은 언어 모델과 번역에 많이 사용되었습니다. 이 같은 순환 모델들은 보통 입출력 시퀀스의 각 심볼 위치에 맞춰 계산을 분할 합니다. 즉 연산 지접을 정렬하여, 이전 은닉 상태 $h_{t-1}$와 $t$ 시점 입력을 함수로 새로운 은닉 $h_{t}$를 생헝합니다.

그러나, 이러한 구조적 특징으로 병렬적으로 학습하기 어렵고, 시퀀스의 길이가 길어질수록 메모리 제약 및 장기 의존성 문제로 긴 예제의 배치를 늘릴기엔 한계가 있습니다.

이러한 문제점을 해결 하기위해, 논문에서는 입출력 시퀀스에서 거리와 무관하게 모델링이 가능한 Attention을 이용해서 순환 구조로를 배제하여 오직 Attention 메커니즘을 사용해서 병렬 처리 및 전연적인 정보를 출력하는 모델인 **Transformer** 모델을 제안합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/0fe81c35-2817-4769-8f1e-fa6a03714d2b" width=400></p>

<p align="center">| Trnasformer Architecture</p>

# 2. Trnasformer

<img src="https://github.com/user-attachments/assets/97644d55-ac8a-491d-a548-fdbfd7f82c70" width=400>

| 간단한 Encoder-Decoder 구조

Transformer는 기존의 seq2seq 모델처럼 인코더-디코더 구조를 가지지만,RNN 없이 오직 어텐션 메커니즘만으로 시퀀스를 처리합니다.

인코더 디코더 각각 하나의 RNN이 시간축 $t$를 가지는 구조 → 인코더 디코더 각각 $N$개의 블록으로 구성


<img src="https://github.com/user-attachments/assets/ee1b7ff8-d704-4dca-aecc-0afecf092fed" width=400>

| 논문에서는 $N=6$으로 6개의 블록 구성

Trnasformer 모델 구조적 위해를 위해 "입력 부분" → "인코더 부분" → "디코더 부분" 으로 단계적으로 확대해서 설명하겠습니다.

## 2.1. Encoder - Decoder

<img src="https://github.com/user-attachments/assets/17ca3adb-ce76-4489-a29c-92fba694fe47" width=400>

| Encoder-Decoder 구조

위에서 서술한 것처럼 Trnasformer는 seq2seq 처럼 인코더 디코더 구조를 가지고 있으며, 인코더에서 정보를 받아 디코더가 결과를 생성합니다.

Trnasformer 디코더는 seq2seq 구조처럼 <sos> → <eos> 출력 까지 연산을 진행 합니다.


## 2.2. Encoder

<img src="https://github.com/user-attachments/assets/7aa222e3-bfea-4ad6-b088-7d504f4a5254" width=200>

| Transformer Encoder

### 2.2.1. Positional Encoding

기존 RNN에서는 단어를 순차적으로 받아서 처리하기 때문에 자연스럽게 단어의 위치정보(position information)를 확인할 수 있었습니다.

 - 순환구조로 앞의 연산이 끝나야 뒤의 연산을 진행할 수 있어서 계산할 유닛이 많아도 한 번에 1개씩 처리한다. 즉, 연산이 느리다.

그러나 Transformer는 단어를 순차적으로 받는 방식이 아니라 전체 시퀀스를 동시에 처리합니다. 이렇게 순서를 무시한 병렬 처리는 단어의 위치정보를 잃어버릴 수 있습니다.

 - Transformer는 압력된 문장을 병렬로 한 번에 처리하는 특징을 가지고 있다. 즉 연산이 빠르다.

논문에서는 이를 해결하기 위해 **Positional Encoding** 을 사용해서 입력 데이터에 순서 정보를 추가합니다.

<details>
<summary>Positional Encoding 톺아보기</summary>

 ### 2.2.1.1. Position Information

---

**"Not"** 의 위치로 인해 문장의 뜻이 달라진다.

### 1. Although I did **"not"** pass the test, I could get a scholarship.
  →  시험에 통과하지 못했지만, 장학금을 받을 수 있었다.

### 2. Although I did pass the test, I could **"not"** get a scholarship.
  → 시험에는 통과했지만, 장학금을 받을 수 없었다.

위치정보는 문장의 문맥을 이해하기에 중요한 정보로, Positional Encoding을 통해 위치정보를 더해줘야 한다.

<img src="https://github.com/user-attachments/assets/61efbf4a-76b3-4f70-b1e2-d89157c1f87e" width=600>

| Positional Encoding + 입력 단어 벡터

---

### 2.2.1.2. Positional Encoding Rules

Positional Encoding에는 2가지 규칙이 있습니다.

### 1) 모든 위치 값은 시퀀스의 길이나 Input에 관계없이 동일한 식별자를 가져야한다.
 시퀀스가 변경되어도 위치 임베딩은 동일하게 유지.

<img src="https://github.com/user-attachments/assets/44e00b49-f556-486f-b397-c75d74ab888f" width=600>


### 2) 모든 위치 값은 너무 크면 안된다.
 위치 값이 너무 커지면, 단어 간의 상관관계를 유추할 수 있는 의미정보가 커져 Attention Layer에서 학습이 안 될 수도 있다.

<img src="https://github.com/user-attachments/assets/b52aab36-4dd5-44c8-a031-6536e131a3b9" width=600>

---

### 2.2.1.3. Positional Encoding with Sine & Cosine 함수

Sine, Cosing 함수는 위의 2가지 규칙을 만족하면서 위치벡터를 추가할 수 있습니다.

### 1) 모든 위치 값은 너무 크면 안된다.
 - Sine & Cosine은 "-1 ~ 1" 사이를 반복하는 주기함수이다. 즉, 1을 초과하지 않으며, -1로 미만으로 가지 않으므로 두 번째 규칙을 지킬 수 있다.

  <img src="https://github.com/user-attachments/assets/e1b7785f-6bde-4324-aea6-86618b733a16" width=600>

### 2) 같은 위치의 토큰은 항상 같은 위치 벡터값을 가지고 있어야 한다.
- Sine &  Cosine 은 주기함수로 -1 ~ 1 사이를 반복하므로 단어 토크들의 위치 벡터 값이 중복 될 수 있다.

   Sine 함수를 예를들어, 단어 토큰이 주어졌을 때, 1 번째 토큰(position 0)과 9 번째 토큰(position 8)의 경우 위치 벡터 값이 같아지는 문제가 발생한다.

  <img src="https://github.com/user-attachments/assets/bac65036-ac22-4eb5-a8a1-bad71b4d10db" width=600>

  하지만 Positional Encoding은 스칼라 값이 아닌 벡터 값으로 단어 벡터와 같은 차원을 지닌 벡터 값이다.

  <img src="https://github.com/user-attachments/assets/0c4d0964-9ced-4714-995c-9fee1f5ccb75" width=600>

  즉, 하나의 위치 벡터가 4개의 차원으로 표현된다면, 각 요소는 서로 다른 4개의 주기를 갖게 되어 서로 겹쳐지지 않는다. 고로 첫 번째 규칙을 지킬 수 있다.
</details>

<img src="https://github.com/user-attachments/assets/104f94c5-30bf-4439-88dc-2e423126040c" width=600>

| Position Encoding이 추가된 Encoder-Decoder

Positional Encoding 값은 아래의 두 함수를 사용합니다.


<p align="center">
$$
PE_{(pos, 2i)} = \sin\left({pos}/{10000^{{2i}/{d_{model}}}}\right)
$$
</p>

<p align="center">
$$
PE_{(pos, 2i+1)} = \cos\left({pos}/{10000^{{2i}/{d_{model}}}}\right)
$$
</p>

$Where:$
  - $pos$ : 입력 문장의 임베딩 벡터의 위치
  - $i$ : 임베딩 베터 내의 차원의 인덱스
  - $d_{model}$ : Transformer의 모든 층의 출력 차원 (Transformer Hyperparameter, 그림에서는 4, But 논문에서는 512)

<p align="left"><img src="https://github.com/user-attachments/assets/8bac064e-36cb-4da5-8492-0fff88be6bd3" width=600>
 <img src="https://github.com/user-attachments/assets/103574bc-89b0-4fc2-aa22-6bbd3a5e1ef7" width=400></p>

임베딩 벡터 내의 각 차원의 인데스가 짝수인 경우 $$(pos, 2i)$$에는 사인 함수를 사용하고, 홀수인 경우$$(pos, 2i + 1)$$에는 코사인 함수를 사용


### 2.2.2 Attention

<img src="https://github.com/user-attachments/assets/16837e94-898c-48e9-a6de-cbb914890068" width=400>

Transforemr에서는 Attention이 어디서 이루어지는지에 떄라 즉, 벡터의 출처가 어디냐에 따라 부르는 이름이 달라집니다.

```
인코더 Self Attention : Query = Key = Value
디코더 Masked Self Attention : Query = Key = Value
디코더 Encdoer-Decoder Attention : Query : 디코더 벡터 / Key = Value : 인코더 벡터
```

<img src="https://github.com/user-attachments/assets/2fa2e6d6-067c-4907-9433-9dc8a577a598" width=600>

