# RAG
 
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks - 2021

ㅡ Patrick Lewis, Ethan Perez

[Paper to Read](https://arxiv.org/pdf/2005.11401)

---

# 1. Introduction

자연어처리(NLP) 분야는 끊임없이 발전하고 있으며, 사전 학습된(Pre-trained) 모델들은 모델의 파라미터 내에 **지식(knowledge)**을 저장하여, 외부 지식에 접근하지 않고도 여러 태스크를 수행할 수 있는 능력을 갖추고 있습니다. 그러나 이 모델들이 저장하는 지식은 업데이트나 확장이 어려운 단점이 있으며, 이로 인해 hallucination(환각) 현상이 발생하기도 합니다.

이러한 문제를 해결하기 위해, 최근의 연구들은 모델이 외부 지식을 적극적으로 활용하여, 복잡한 질문에 더 정확한 답을 할 수 있도록 하는 방법에 초점을 맞추고 있습니다.

즉, 기존의 pre-trained 모델이 가지고 있는 parametric knowledge와 외부에서 검색된 non-parametric knowledge를 함께 활용하는 모델들이 연구되었으며, 그 예로 REALM이나 ORQA와 같은 retrieval-augmented language models가 있습니다.

이러한 retrieval-augmented language models는 open-domain extractive question answering (QA) 태스크에서 좋은 성과를 보였으나, BERT와 같은 Encoder-only 모델을 기반으로 하여 특정 조건 하에서만 open-QA 태스크를 수행할 수 있는 한계가 있습니다.

이런 배경 속에서 “Retrieval-Augmented Generation”논문은 기존 언어 모델의 한계를 해결하고, 외부 지식을 효과적으로 활용하고 답을 생성하는 새로운 방법론을 재시합니다.

RAG 모델은 Retrieval과 Generator 부분으로 이루어져 있습니다. Retreiver는 BERT와 같은 강력한 언어 모델을 사용해서 문서(Document)와 질문(Query)를 벡터 공간에 임베딩하고, 이를 통해 질문과 관련된 정보를 검색합니다.

그런 다음, 검색된 정보는 Vector Concatenation을 통해 하나의 벡터로 결합되어, Generator의 입력으로 사용됩니다. Generator는 이 정보를 바탕으로 최종적인 답을 생성합니다. 이 과정은 Supervised Loss를 통해 학습되며, $BERT_q(Query)$ 와 Genrator만 학습해도 문서를 어떻게 잘 검색할지 자동으로 최적화 됩니다. RAG는 문서 Index가 존재하긴 하지만, 그 인덱스를 어떻게 활용해야하는지까지 통합적으로 학습되기 때문에 효과적으로 학습할 수 있습니다.

# 2. Method

<img src="https://github.com/user-attachments/assets/6160effe-056a-4e2a-9f42-66dafff8ce54" width=700>

## 2.1. Retriever

논문에서 제일 먼저 살펴봐야하는 부분은 ‘Retriver’입니다. 검색기로, RAG 모델에서 중요한 역할을 담당합니다. 

Retriever를 설명하기 앞서 질문(Query)와 문서(Document)에 대해서 알아볼 필요가 있습니다.

### 2.1.1. Query

Query 는 모델이 답변을 해야 할 질문입니다. 예를들어, “프랑스의 수도는 어디야?”와 같은 질문입니다.

모델은 이 Query를 BERT와 같은 언어 모델을 사용해서 벡터화합니다. 이 벡터는 질문의 의미를 고차원 공간에 임베딩해서, 문서(Document)와의 관련성을 계산하는 데 사용됩니다.

### 2.1.2. Document

Document는 질문에 대한 정보를 포함하고 있는 문서입니다. Document는 Wikipedia와 같은 대규모 텍스트 집합일 수 있습니다. 모델은 이 문서들을 벡터화 해서, 각 문서가 Query와 얼마나 관련이 있는지 평가합니다. 즉, 문서 내의 내용이 Query와 얼마나 일치하는지를 파악해서, 관련이 높은 문서를 선택해서 사용합니다.

### 2.2.3. Query와 Document의 활용

위의 “프랑스의 수도는 어디야”라는 질문(Query)이 모델에 들어왔을때, 기존 Model은 이러한 정보를 학습한 적이 없다고 가정해보겠습니다. 여기서, Model은 질문(Query)에 대한 답변을 하기 위해 필요한 정보를 얻어야 합니다. 그 정보를 담고 있는 것이 Document 입니다. 그러나 Document에 포함된 수많은 정보 중에서 질문(Query)에 맞는 대답을 하기 위해서는 알 맞는 정보들만 골라야합니다.  그 작업을 검색(Retriever)이라 불리며, 논문에서는 Retriever 부릅니다. 



### 2.2.4. DPR(Dense Passage Retriever)

RAG에서는 DPR이라는 방식으로 Retriever를 진행합니다. 질문과 관련된 문서를 효과적으로 검색하는데 사용되며, BERT와 같은 사전 학습된 언어 모델을 활용합니다. 모델이 받은 질문과 문서를 각각 베터화 하고, 이 벡터들 간의 유사도를 계산해서 관련성이 높은 문서를 검색합니다.

**BI-Encoder**

DPR은 BI-Encoder를 사용합니다, 즉, 질문(Query)와 문서(Document)는 질문 인코더($BERT_{q}$), 문서 인코더($BERT_{d}$)로 나뉘게 되며 각각의 인코더를 통해서 벡터화( $q(x)$, $d(x)$ ) 됩니다. 각 인코더는 BERT와 같은 Transformer 모델을 기반으로 합니다.

$$
Query → BERT_{q} → q(x)
$$

$$
Document → BERT_{d} → d(x)
$$

벡터화가 된 질문 벡터( $q(x)$ )와 문서 벡터( $d(x)$ )는 MIPS(Maximum Inner Product Search)를 사용해서 유사도를 계산합니다.

**Cosine Similarity vs MIPS**
    
### Cosine Similarity
    
Cosine Similarity는 두 벡터의 내적을 벡터의 크기로 나누어, 벡터 방향에 대한 유사도를 평가합니다. 따라서 벡터의 크기가 유사도 계산의 영향을 미칩니다.
    
$$Cosine Similarity = \frac{q \cdot d}{|q||d|}$$
    
$Where$
    
- $q$ : Query Vector
- $d$ : Document Vector
- $\cdot$ : 내적
- $|q|, |d|$ : Query의 크기, Document의 크기
    
### MIPS (Maximum Inner Product Search)
    
MIPS는 내적만 계산하고, 벡터의 크기에 대해서는 고려하지 않습니다. 벡터의 크기를 고려하지 않기 때문에, 정규화된 벡터가 필요하지 않습니다. 대신에 최대 내적값을 계산해서 가장 유사한 문서를 선택합니다. 즉, Cosine Similarity 보다 연산량이 작습니다.
    
이로 인해, 대규모 문서 집합에서 가장 관련성이 높은 문서를 빠르게 찾는 데 유리해집니다. 
    
$$MIPS = \underset{i \in S}{arg\, max} \ \langle x_i, q \rangle$$
    
$Where$
    
- $S$ : 데이터가 저장된 벡터 공간 집합
- $x_i$ : $i$번째 Data Vector
- $q$ : Query Vector
- $\langle x_i, q \rangle$ : Data Vector $x_i$와 Query Vector 내적
- $\underset{i \in S}{arg\, max}$ : $i$값을 찾는 연산자로, Data 집합 $S$에 속한 벡터 $x_i$ 중에서 $q$(Query Vector) 와 내적이 가장 큰 Vector 값을 찾는 연산자.

계산된 MIPS 값을 기준으로 가장 유사한 상위 k개의 문서를 선택합니다. 이 상위 k개의 문서는 질문에 가장 관련이 높은 문서들입니다.

<img src="https://github.com/user-attachments/assets/882eef25-b205-4673-bcbb-14c5286ea5ec" width=500>

## 2.2. Vector Concatenation

선택한 상위 k개의 문서 즉, Document Vector는 Query Vector와 Concat하여 하나의 벡터로 만듭니다.

k개의 문서에 대해 각 문서와 쿼리를 각각 결합하므로, 총 k개의 벡터 덩어리가 생성됩니다.

$$
V_i = concat(q,x_i)
$$

$Where$

- $q$ : Query Vector
- $x_i$ : $i$번째 Document Vector

다만 실제 구현에서는 문자열 형태의 질의와 문서를 그대로 이어붙여 하나의 텍스트 시퀀스를 만들고, 이를 BART(Generator)의 인코더에 입력합니다. 

이렇게 생성된 k개의 시퀀스 각각에 대해 BART 디코더가 답변 후보를 생성하며, 학습 과정에서는 Negative Log-Likelihood(Loss)를 계산하여 가장 적합한 문서-질문 쌍(또는 토큰 시점별 문서)을 선택하거나 확률을 marginalize하는 식으로 최종 답변을 결정합니다.

학습 시에는 여러 문서 중에서 실제로 Loss가 가장 낮아지는(즉, 타깃 답변과 가장 잘 맞아떨어지는) 문서를 선택하여 답을 생성하는 방식으로 최적화됩니다.

<img src="https://github.com/user-attachments/assets/4c97680e-e908-4691-974d-dde31e5202d8" width=700>

Query와 가까운 Top-5 Document(Passage)를 이용해서 각 Document와 Query를 Concat하여 BART Input을 생성합니다.
 (하나의 Query 당 5개의 BART Input이 존재하며, 배치처리를 통해 추론하여 빠른 속도로 처리가 가능합니다.)

각 (Document, Query)에서 Vocab Distribution을 생성하며 Aggregation(Marginalization) 과정을 통해 하나의 문장을 생성합니다.


## 2.3. Generator

Generator는 문서와 쿼리를 결합한 벡터를 입력받고, 그 정보로 부터 답변을 생성하는 역할을 합니다.

Rag에서 사용하는 Generator는 일반적으로 **seq2seq (Sequence-to-Sequence)** 모델을 사용하며, 그 중 하나가 바로 **BART**입니다.

### 2.3.1. Generator의 구조

**Generator**의 구조는 **Encoder-Decoder** 아키텍처로 되어 있으며, **BART** 모델을 활용하여 각 단계가 처리됩니다.

### 2.3.1.1. BART (Bidirectional and Auto-Regressive Tansformers)

BART는 Encoder-Decoder 구조를 기반으로, 양방향 인코딩과 자동회귀 디코딩 방식을 결합한 모델입니다. BART는 BERT와 GPT의 장점을 결합하여, 입력 문장을 이해하고, 자연스러운 답변을 생성할 수 있습니다.

BART Encoder는 BERT 처럼 양방향문맥을 이해하고, BART Decoder는 GPT처럼 자동 회귀 방식으로 단어를 생성합니다. 

### 2.3.1.2. Encoder

Retriever에서 나와서 결합된 문서와 쿼리 벡터는 Generator의 Encoder에 먼저 인코딩됩니다. 이 단계에서는 각 벡터를 Transformer 네트워크를 통해 contextualized하게 인코딩 됩니다.

BART의 Encoder는 양방향처리(Bidirectional) 방식으로 작동되며, 문서와 쿼리의 양방향 문맥을 고려라여 벡터를 인코딩합니다. 즉, 쿼리와 문서에서 주어진 단어들이 문맥에 맞게 각기 다른 의미로 인코딩 됩니다.

### 2.3.1.3. Decoder

Generator의 Decoder는 Encoder에서 나온 벡터를 바탕으로 답변을 생성합니다.

이때 자연어 텍스트는 자동 회귀 방식(Auto-regressive)으로 순차적으로 생성하며, 모델이 생성하는 단어들은 이전 단어와 문맥을 고려하여 결정됩니다.

BART의 Decoder는 Self-attention을 사용하여 각 단어가 문맥에 맞는 가장 적합한 단어로 생성되도록 합니다.



### 2.3.1.4. Decoding

RAG 모델은 디코딩을 RAG-Token과 RAG-Sequence 두 가지 방식을 선택적으로 사용합니다. 두 모델은 **디코딩 방식**에 차이가 있으며, 학습과 출력 과정에서의 차이로 인해 성능 차이를 보일 수 있습니다.

### 3. Beam Search
<img src="https://github.com/user-attachments/assets/4c64a47b-784d-45f7-a92f-cca905d5e298" width=600>

Beam Search는 Greedy Search와 다르게, 모든 가능한 경우의 수를 탐색하는 대신, 최적의 후보를 선택하여 탐색을 진행하는 방식입니다. 일반적으로 **Beam Width (n)** 를 설정하여, 각 단계에서 n개의 후보를 선택하고, 그 후보들 중에서 최적의 시퀀스를 찾는 방식입니다.

Greedy Search는 각 단계에서 확률값이 가장 높은 토큰을 선택하여 문장을 생성하지만, 이는 항상 최선의 문장을 생성하지는 않는다는 문제가 있습니다.

Beam Search는 이 문제를 해결하기 위해, 각 단계에서 n개의 토큰 후보를 고려하여, 여러 가지 가능한 시퀀스를 생성합니다. 위 그림은 $n=3$을 예시로 한 그림입니다.

Start 토큰을 기점으로 가장 확률값이 높은 3개의 토큰을 선택합니다. 'a', 'I', 'the'가 선택된 토큰이 됩니다.

선택된 각 토큰에서 다음에 올 토큰을 예측합니다. 각 토큰에 대해 다음에 올 수 있는 토큰들을 3개씩 뽑아, 총 9개의 후보를 만듭니다.

그 후, 각 후보 시퀀스의 확률값을 평가하여 가장 높은 확률을 가진 3개의 시퀀스를 선택합니다. 이렇게 선택된 시퀀스는 'a cat', 'I am', 'the dog' 등이 될 수 있습니다.

여러 번의 확장을 거쳐, 최종적으로 가장 높은 확률을 가진 시퀀스를 선택하게 됩니다.

Greedy Search vs Beam Search:

Greedy Search: 매 분기마다 확률이 가장 높은 단일 토큰을 선택합니다. 초기 선택이 최선일 수 있으나, 전체적으로 최선의 결과를 만들지 못할 수 있습니다.

예시 문장: "a cat meowed"

Beam Search: 여러 후보 시퀀스를 고려하여, 그 중 가장 높은 확률을 가진 시퀀스를 선택합니다. 초기 선택이 최선이 아닐지라도, 여러 후보를 고려함으로써 보다 높은 확률을 가진 최종 결과를 얻을 수 있습니다.

예시 문장: "the dog barked suddenly"

Beam Search는 Greedy Search보다 더 다양한 후보를 고려하며, 결과적으로 더 높은 확률의 시퀀스를 생성할 수 있는 방법입니다.

### 2. **RAG-Token Model**
    
   RAG-Token 모델은 각 토큰에 대해 다른 문서를 선택하여 답변을 생성합니다.
    
   Generator는 각 토큰마다 다른 문서를 사용하여 답변을 생성합니다. 이는 각 출력 토큰을 생성할 때마다 관련 문서를 동적으로 선택하는 방식입니다.
    
   이 모델은 다양한 문서들에서 각 토큰을 생성하며, 마진화된 결과로 최종적으로 답변을 완성합니다.

   여러 개의 문서를 동적으로 활용할 수 있기에, 다양한 정보 소스를 바탕으로 답변을 생성할 수 있습니다. 그러나 여러 문서를 처리하기 때문에 복잡하고 계산량이 커질 수 있습니다.

   또한 이 방법은 기존 beam search 방법론에도 그대로 활용이 가능합니다.
    
$$pRAG-Token(y∣x)≈∏_{i=1}^N∑_{z∈top-k}p_η(z∣x)p_θ(y_i∣x,z,y_{1:i−1})$$

$Wherer$

- $p_η : Retrieval$

- $p_θ : Generator$

 <img src="https://github.com/user-attachments/assets/8c381788-0b70-421c-8a66-0ed084148951" width=700>


### 3. **RAG-Sequence Model**
    
   RAG-Sequence 모델은 전체 시퀀스에 대해 하나의 문서만을 사용하여 답변을 생성합니다.
    
   Generator는 전체 시퀀스를 하나의 문서에 기반하여 생성하며, 문서 하나에 대해 전체 시퀀스를 생성하는 방식입니다.
    
   질문에 대한 단일 정보 소스로 부터, 그 문서를 기준으로 답변을 생성하므로, 문백을 잘 반영할 수 있습니다. 그러나 하나의 문서만을 사용하는 방식이어서 다양한 문서를 고려할 수 없는 단점이 있습니다.

    
$$pRAG-Sequence(y∣x)≈∑_{z∈top-k}p_η(z∣x)p_θ(y∣x,z)$$
    
$$= ∑_{z∈top-k}p_η(z∣x)∏_i^Np_θ(y_i∣x,z,y_{1:i-1})$$

$Wherer$

- $p_η : Retrieval$

- $p_θ : Generator$

이러한 점에서 기존 Beam Search를 적용하기에 문제가 있어서 논문에서는 document 별로 beam search를 진행합니다.

<img src="https://github.com/user-attachments/assets/e2f8b8e6-bafb-49a3-b95f-b0af02d1ce62" width=700>

입력 $x$에 대해 **top-k문서** 를 검색합니다. 여기서는 $k=3$으로 설정되어 있어, 3개의 문서가 선택됩니다.

선택된 각 문서 $z$에 대해 Beam Search를 진행하여, 해당 문서에서 최적의 sequence를 생성합니다. 이때, 문서별로 $k$ 개의 후보가 선택됩니다. 여기서의 $k$는 **top-k** 와는 별개로, 각 문서에 대해 독립적인 Beam Search가 적용됩니다.

각 문서 $z$에 대해 구해진 logit 순서에 따라 확률이 계산됩니다. $p(y|x, z)$와 같이 문서별로 출력 $y$에 대한 확률이 계산됩니다.

각 문서에서 구해진 확률을 marginalize하여 최종 확률 $p(y|x)$를 구합니다. 즉, 모든 문서에 대한 $p(y|x, z)$ 값을 더해 최종 출력 확률을 계산합니다.

<img src="https://github.com/user-attachments/assets/df95640e-6923-4b10-b061-89dbcd41107b" wdith=700>

그러나, 만약 Beam Search 과정에서 특정 값이 계산되지 않으면, 해당 확률 값을 알 수 없는 문제가 발생합니다. 그림처럼, $y_2$는 문서 $z_2$로부터 계산되지 않았는데, 이를 marginalize 하려면 $p(y_2|x, z_2)$ 값을 알아야 합니다. 하지만 이 값을 알 수 없어서 계산이 불가능한 상황이 발생합니다.

이 문제를 해결하기 위해 추가적인 forward pass를 통해 $p(y_2|x, z_2)$ 값을 계산해야 합니다. 하지만 이 과정은 비효율적이고 시간적, 비용적 부담이 크기 때문에 적합하지 않습니다.

<img src="https://github.com/user-attachments/assets/2fdcec7c-8691-4ec9-abe0-7ce86b2be5d0" width=700>

논문에서는 이를 해결하기 위해, 발견되지 않은 값들에 대해 확률을 0으로 처리하는 방법인 **Fast Decoding**을 제시합니다. 이 방식으로 추가적인 계산을 피하면서 marginalize를 진행할 수 있게 해줍니다.

## 2.4 정리
<img src="https://github.com/user-attachments/assets/687dd9bf-1013-4f17-804a-925690ffd1c4" width=700>

Input sequence인 $x$를 input으로 받아서, 이를 활용하여 external knowledge corpus(말뭉치)로 관련이 제일 높은 document $z$를 $k$개 retreiver 합니다. 이후 $x$와 $d$를 이용해서 target sequence인 $y$를 생성하면서 두 가지 요소를 다루게 됩니다.

retriever $p_η(z|x)$ with parameter $η$: $x$에 대한 document의 distribution

generator $p_θ(yi|x,z,y1:i−1)$ parametrized by $θ$ : $x$와 $z$를 바탕으로 $y$생성

두 요소들은 훈련과정에서 동시에 학습되며 output 결괴를 위해 검색된 document에 대해 Marginate를 진행하게 됩니다. RAG 논문에서는 RAG-Sequence와 RAG-Token을 제안했습니다.


## 2.5 Training

RAG 모델의 학습과정은 Retriever과 Generator를 동시에 학습시키는 End-to-End 방식입니다. 문서 검색과 응답 생성을 함께 최적화하여 Loss를 계산합니다.

중요한 점은 **어떤 문서를 검색해야 할지에 대한 직접적인 지도(supervision)가 없다는 것**입니다. 즉, **문서 검색**과 **응답 생성**이 함께 최적화됩니다. 

### 2.5.1. Training

입력/출력 쌍 $(x_j, y_j)$가 주어집니다. 여기서 $x_j$는 입력 Query이고, $y_j$는 출력 답변입니다.

각 타겟에 대한 Negative marginal log-likelhood를 최소화해야합니다. 이를 통해 RAG모델이 적절한 문서를 검색하고 정확한 답변을 생성하도록 학습합니다.

$$
Loss = -\sum_j{log p(y_j|x_j)}
$$

$p(y_j|x_j)$는 입력 쿼리 $x_j$에 대해 정확한 답변 $y_j$를 생성할 확률로, 이 확률을 최소화 하는 방향으로 학습합니다.

### 2.5.2. Retriever 학습

Retriever는 쿼리 $x_j$에 대해 가장 관련성이 높은 문서를 검색합니다. 하지만, 어떤 문서를 검색해야 하는지에 대한 지도학습이 없습니다. 대신  Document Encoder인 $BERT_d$를 동결하고, Query Encoder $BERT_q$와 Generator인 $BART$만 fine-tuning 합니다.

### 2.5.3. Document Encoder Frozen

Document Encoder $BERT_d$는 학습 중에 업데이트하지 않습니다. 그 이유는 문서 인데스를 주기적으로 업데이트 하기 때문입니다. 이전 연구인 "REALM" 모델처럼 문서 인코더를 매번 갱신하며 학습을 진행했는데, 그 비용이 매우 큰 단점이 있었습니다. RAG 모델에서는 문서 인코더 $BERT_d$를 동결시키고 쿼리 인코더와 BART만 학습시켜 문제를 크게 완화했습니다.

## 2.6 Experiments
Wikipedia 2018년 12월 덤프를 사용하였으며 Wikipedia article은 100-word로 분할하여 총 21,015,324개 document가 된다.
DPR document retriever를 사용하여 각 document에 대한 document embedding을 계산하고 효율적인 검색을 위해 Hierarchical Navigable Small World approximation를 사용하는 FAISS를 통해 single MIPS index를 만든다.

top-k개의 document를 검색하며 상위 k는 {5,10}으로 세팅하여 진행했습니다.


[출처 : FAISS Github ](https://github.com/facebookresearch/faiss)

### IndexFlatIP 생성
```c
int faiss_IndexFlatIP_new(FaissIndexFlatIP** p_index) {
    try {
        IndexFlatIP* index = new IndexFlatIP(); // IndexFlatIP 생성 (내적 계산용 인덱스)
        *p_index = reinterpret_cast<FaissIndexFlatIP*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexFlatIP_new_with(FaissIndexFlatIP** p_index, idx_t d) {
    try {
        IndexFlatIP* index = new IndexFlatIP(d); // 벡터 차원을 지정하여 IndexFlatIP 생성
        *p_index = reinterpret_cast<FaissIndexFlatIP*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}
```

### MIPS 검색
```c
int faiss_IndexFlat_compute_distance_subset(
        FaissIndex* index,          // Faiss 인덱스 객체
        idx_t n,                    // 쿼리 벡터의 개수
        const float* x,             // 쿼리 벡터들
        idx_t k,                    // 검색할 유사한 벡터의 개수
        float* distances,           // 유사도 결과를 저장할 배열
        const idx_t* labels) {      // 각 벡터에 대한 레이블/인덱스 배열

    try {
        // 실제로 거리를 계산하는 함수
        reinterpret_cast<IndexFlat*>(index)->compute_distance_subset(
                n, x, k, distances, labels);
        return 0;
    } CATCH_AND_HANDLE
}
```

![image](https://github.com/user-attachments/assets/585b9d2c-ebf2-48de-b6c6-1c7932bede76)

Left & Cneter
- RAG-SEQ의 경우 Retreived Documents의 수가 많을 수록 성능이 개선
- RAG-TOK의 경우 Retreived Documents의 수가 일정 수 이상이면 성능이 저하

Right
- Abstractive Answer Generation의 경우 많은 Documents가 필요하지 않음(less than 10)
(제한된 외부 지식을 효과적으로 활용)

그래서 Top-k를 5개에서 10개 사이로 실험을 진행 했습니다.

## 2.7 Conclusion

1. Retreiver과 Generator를 결합
   - Retreiever에 대해 End-to-End로 학습을 진행항여 성능을 개선
   - Generator를 사용 (Diversity, Hallucination 해결)
     
2. 외부 지식을 사용
   - 언어 모델의 Knowledge Base 수정 가능
   - 언어 모델의 생성 결과에 대한 해석 가능


