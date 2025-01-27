# Vision_Transformer
AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

-Alexey Dosovitskiy, Lucas Beyer

Google Research, Brain Team
 
https://arxiv.org/pdf/2010.11929

---

# 1. Introduction

Self-Attention 기반 아키텍처를 활용한 Transformer는 NLP 분야에서 선두 주자로 자리매김 했었습니다. Transformer의 계산 효율성 및 확장석 덕분에 1,000억 개가 넘는 파라미터를 가진 모델들도 학습이 가능해졌으며, 모델 및 데이터셋이 점점 커져가면서 성능 포화 현상이 이루어지 않고 지속적으로 개선이 가능해졌습니다.
또한 대규모 데이터셋으로 사전 학습한 후, 작은 데이터셋으로 다운스트림 작업을 수행할 수 있어 효율적인 학습이 가능합니다.

이 논문은 NLP에서 큰 성공을 거둔 Transformer를 활용해서 컴퓨터 비전으로 확장하는 방안을 제시했습니다. 이미지 데이터를 패치로 분할하고, 패치들을 선형 임베딩 즉 1D 시퀀스로 변환하여 Transformer로 학습합니다.

그러나, 다양한 정규화 기법 없이 ImageNet과 같은 중간 규모 데이터셋으로 훈련했을 때 Transformer는 기존의 CNN 기반 모델인 ResNet 보다 낮은 정확도를 보여줍니다. 이는 Transformer가 CNN에서 제공하는 Locality 및 Translation Equivariance와 같은 Inductive Bias(귀납적 편향)가 부족하기 때문입니다. 이런 귀납적 편향은 상대적으로 적은 데이터 셋에서는 일반화하기가 유리하지만, Transforem 는 귀납적 편향을 포함하고 있지 않기 때문에 대규모 데이터셋에서 학습이 필요합니다.

Vision Transformer(ViT)는 규모가 큰 데이터셋(ex, JFT-300M, ImageNet-21k)으로 학습하면 ResNet 보다 더 뛰어난 성능을 발휘할 수 있습니다. 이 말은 위에서 언급한 ViT가 CNN과 다르게 모델의 규모가 커지고 데이터셋 크기가 증가해도 성능 포화 되지 않고 지속적으로 향상되는 특징을 가지기 때문입니다.


# Related Work

## 2.1. 기존 Transformer의 이미지 처리 접근법

이전에는 이미지 처리에 Transformer를 적용하는 다양한 방법들이 시도되었습니다.

1. Self-Attentino을 이미지에 적용
   -  모든 픽셀 간의 상호작용을 고려해야 하며, **$$O(N^2)$$** 의 시간 복잡도를 가짐
 
2. 각 쿼리 픽셀에 Local 영역 내에서 Self-Attention을 적용
   - Global 영역을 대상으로 하지 않고 Local Self-Attention을 적용

3. Sparse Transformer
   - Global Self-Attention을 효율적으로 근사하여 이미지 처리

이런 방식은 하드웨어 가속기에서 연산을 효율적으로 수행하기에는 다소 번거로운 작업이며, 더 나은 성능을 위해 여러가지 최적화가 필요했습니다.

## 2.2. ViT 유사 접근법

1. 이미지를 2X2의 패치로 쪼개서 Self-Attention을 적용
   - 이 방식은 작은 패치 크기를 사용하여 해상도가 작은 이미지에만 적용 가능

2. 이미지 해상도 및 새강고오간 축소후 픽셀단위로 Transformer 적용
   - iGPT(Image GPT) 모델로 비지도 학습 방식으로 훈련되며, ImageNet에서 최대 72%의 정확도를 달성

## 2.3. Vision Transformer

ViT는 기존의 Transformer 모델들이 가진 한계를 극복하기 위해, 이미지 데이터를 16X16, 32X32와 같은 상대적으로 큰 패치 크기로 분할하여 Self-Attention을 적용합니다. 그로인해 이미지의 각 부분에 대한 정보를 보다 효과적으로 캡쳐할 수 있도록 도와주며, 높은 해상도의 이미지 처리를 가능하게 해줍니다.

결론적으로 ViT를 표준 ImageNet 데이터셋 보다 더 큰 크기의 데이터 셋에서 Image Recognition 실험을 진행하였고, 더 큰 크기의 데이터 셋으로 학습시켜서 기존 ResNet 기반 CNN 모델모다 더 좋은 성능을 내는 Vision Transformer 모델을 만들 수 있었습니다.

# 3. Architecture

<img src="https://github.com/user-attachments/assets/1db9cbe3-324c-4dfd-ade8-4011bee04c7e" width=800>

## 3.1. Patch Embedding
Transformer는 1D 시퀀스를 입력으로 받기 때문에, ViT에서는 이미지를 $$2D$$ 형태에서 $$1D$$ 시퀀스로 변환해야합니다. 이미지는 원래 $$x \in \mathbb{R}^{H \times W \times C}$$ 형태에서, $$P \times P$$ 크기의 작은 패치들로 나눕니다.


$$ x \in \mathbb{R}^{H \times W \times C} $$

하나의 이미지가 $P \times P$ 크기의 패치들로 나누어진 후, 각 패치는 $P^2 \times C$ 크기의 벡터로 변환됩니다. 이때, 패치의 개수 $N$은 다음과 같이 정의됩니다

$$ x_p \in \mathbb{R}^{N \times (P^2 \times C)}, \quad N = \frac{H \times W}{P^2} $$

$Where:$
- $H$ : 이미지 높이
- $W$ : 이미지 너비
- $C$ : 채널 수
- $P$ : 패치 크기
- $N$ : 패치 갯수

이렇게 나눈 패치는 $$N \times (P^2⋅C)$$ 크기의 $$1D$$ 시퀀스로 변환되어 Transformer의 입력값으로 들어갑니다.

<img src="https://github.com/user-attachments/assets/628c82f5-ca66-42b8-a4b8-e15f9f577947" width=600>

따라서 $1D$로 변환되는 시퀀스의 계산은 다음과 같습니다.

$Input Img(H \times W) = 224 X 224$

$Patch = 16 \times 16$

$Channel = 3(RGB)$

$N = \frac{224 \times 224}{16^2} = 256$

$Final Sequence = 256 \times (14^2 \times 3) = 150,528$

Input Img 224를 크기가 16인 패치로 나누면 가로 세로 14개로 196개의 패치가 생기고, 가로세로 16인 각 패치가 총 3개의 채널 (RGB)를 가지고 있기에, 그 식은 다음과 같습니다.

$16 \times 16 \times 3 \times 14 \times 14 = 150,528$

## 3.2. [CLS]token & Position Embeddings
## 3.2.1. [CLS]token
ViT 모델은 Image Classification을 위해 [CLS] = Classification 토큰을 추가합니다. 이 토큰은 앞서 만든 Patch Embedding 앞에 위치하며, Transformer에서 출력할때 최종 이미지를 결정하는 정보를 추출합니다.

<img src="https://github.com/user-attachments/assets/2cca2a31-ec89-4e78-ab7a-b4e8cad93604" width=500>


[CLS]token은 그림에서 파란색 부분으로 그 크기는 $1 \times 768$ 텐서입니다. (ViT의 hidden size)

이 [CLS]token은 Patch Embedding 맨 앞에 추가되며, 그림처럼 패치 수가 196개라면, Transformer 입력 시퀀스의 크기는 $197 \times 768$ (196개의 패치 + 1개 class token)

## 3.2.2. Positinoal Embedding
Transformer 모델은 기본적으로 입력 시퀀스의 순서에 대한 정보를 스스로 처리할 수 없습니다. 그래서 **Positional Embedding**을 사용하여 입력 시퀀스에 각 위치 정보를 부여합니다. 이 정보는 모델이 시퀀스 내의 요소들 간의 상대적인 위치 관계를 학습하는 데 도움을 줍니다.

<img src="https://github.com/user-attachments/assets/270d4bd2-a55c-45fd-88e9-6c4d19339f55" width=700>

Positional Embedding은 학습 가능한 $197 \times 768$ 텐서로, **1D** 형태입니다. 여기서 **197**은 이미지 패치 196개와 **[CLS] token**을 포함한 총 시퀀스 길이를 의미하며, **768**은 ViT 모델의 hidden size와 동일합니다.
 
ViT에서는 **2D Positional Embedding** 대신 **1D Positional Embedding**을 사용하지만, 성능 차이가 크게 나지 않았습니다. 이는 이미지를 패치로 나누는 과정에서 공간적 정보가 상대적으로 덜 중요해지기 때문입니다. 

<img src="https://github.com/user-attachments/assets/e20cc90f-1ca8-40ae-a895-2ee564017e97" width=1100>

**Positional Embedding**은 **[CLS] token**과 결합되어 최종적으로 입력 시퀀스를 형성하고, 이를 Transformer에 입력하여 이미지를 처리합니다. 이를 통해 모델은 각 패치의 상대적인 위치를 학습할 수 있게 되어 Img Classification 작업에 도움을 줍니다.


## 3.2.3. Process
<img src="https://github.com/user-attachments/assets/3879a73e-a89d-49c3-b569-353e19e071ff" width=1100>


# 3.3. Transformer Encoder
ViT의 핵심부분인 Transformer Encoder 입니다. Transformer Encoder는 여러 개의 Self-Attentino 및 MLP (FFNN) Blcok이 번갈아 쌓인 구조로 이루어져 있습니다.

- Self-Attention : 각 Patch가 다른 Patch들과 어떻게 관련이 있는지를 계산하는 메커니즘
<details>
  <summary>Self-Attention</summary>
  
1. Image Patch
   
[CLS]token & Position Embeddings 과정이 끝난 $1D$ 시퀀스를 Transformer 모델에 입력으로 받습니다.

2. Q, K, V Vector
   
각 Patch Embedding은 학습 가능한 가중치 행렬을 곱해 Query(Q), Key(K), Value(V) 벡터를 생성합니다. QKV 벡터를 활용하여 Self-Attention을 계산하는데, Q 벡터와 K 벡터와 내적을 통해 Attention Score를 계산하고, 이 Score는 $\sqrt{d_k}$ (K 벡터 차원)에 대해 스케일 후, Softmax로 Attention Distribution을 생성합니다.
     
   $$Attention(Q,K,V) = Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

3. Multi-head Attention
   
ViT는 여러 개의 Attention Head를 병렬로 사용하는 Multi-head Attention을 사용합니다. 기존 Self-Attention 연산을 여러 개를 한 번에 처리해서 패치 간의 어텐션 연산을 병렬 처리로 수행하게끔 합니다. 각 헤드는 서로 다른 부분들을 집중할 수 있어 다양한 관계를 학습할 수 있습니다.
</details>
   
    
- MLP : Self-Attention 계산 후에 비선형 변환을 수행하는 층으로, 모델의 표현력을 높입니다.

<details>
   <summary>MLP</summary>
1. MLP
     
MLP는 Transformer 내에서 Self-Attention으로 얻은 패치들 간의 관계 정보를 비선형적으로 변환하는 역할을 합니다. MLP가 포함된 Block은 정보 간의 관계를 확장 및 더 정교한 표현을 학습하는데 도움을 줍니다.

2. MLP 수식

MLP는 두 개의 선형 계층을 포함하며, 각 계층 후에 활성화함수(GELU)가 적용됩니다.

  $$z_1 = W_1x + b_1$$

  $$z_2 = GELU(z_1)$$

  $$z_3 = W_2z_2 + b_2$$
  
3. MLP의 장점
  
- 비선형성: 활성화 함수를 통해 비선형 변환을 추가함으로써 모델의 표현 능력을 향상시킵니다.
- 연산 효율성: 선형 변환과 활성화 함수만을 사용하여 효율적인 계산을 할 수 있습니다.
- 정보 확장: Self-Attention에서 얻은 정보를 더 높은 차원으로 변환하여, 모델이 더 복잡한 관계를 학습할 수 있도록 합니다.
 </details>

    
- LayerNorm : 각 Block에 대한 정규화 기법으로, 학습의 안정성을 높이고 성능을 개선합니다.

<details>
<summary>Layer Normalization</summary>
    
LayerNorm은 활성화 배치의 각 항목의 평균과 분산을 계산하여 이를 사용해 데이터를 정규화하는 기법입니다.
    
기존 Batch Normalization에서 Layer Normalization으로 변경하는데, Batch가 많아야지 적용이 잘되는 Batch Normalization인데, 데이터가 너무 많아 배치 크기를 줄이는 경우가 많습니다.
    
그로 인해 배치 크기가 작아져 평균과 분산 추정이 부정확해져 성능이 저하됩니다.
    
입력 데이터의 모양이 $[N,C,H,W]$일 때, 각 배치 내에서 $[C,H,W]$ 모양의 모든 요소에 대해 평균과 분산을 계산합니다.
    
배치 크기에 의존하지 않아 배치 정규화에서 생기는 문제를 해결하며, 평균과 분산 값을 따로 저장할 필요가 없습니다.
    
| **특징**           | **BatchNorm**                     | **LayerNorm**                 |
|--------------------|-----------------------------------|-------------------------------|
| **평균/분산 계산 기준** | 배치 단위                        | 각 샘플(feature 차원)         |
| **배치 크기 의존성** | 크면 안정적, 작으면 성능 저하      | 독립적, 소규모 배치에서도 안정 |
| **순차 데이터 처리**  | 비효율적                         | 적합                          |
| **추론 단계**       | 평균/분산 저장 필요              | 추가 저장 필요 없음            |
    
<img src="https://github.com/user-attachments/assets/40930afb-50c9-4a99-9fd2-5b630d39b8e3" width=600>
    
층 정규화 수식은 다음과 같습니다.
벡터 $h$의 평균 $\mu$와 분산 $\sigma^2$을 계산한 후, 각 차원 $h_k$의 값을 아래 수식으로 정규화합니다.
    
여기서 $ϵ$ 은 분모가 0이 되는 것을 방지하기 위한 작은 값입니다.
    
$$x̂ᵢₖ = (xᵢₖ - μᵢ) / √{(σᵢ² + ε)}$$
    
정규화된 값에 학습 가능한 파라미터 $\gamma$와 $\beta$를 적용하여 최종 정규화 값을 계산합니다.
    
$$y_k = \gamma \hat{h}_k + \beta$$

$\gamma$와 $\beta$는 초기값으로 각각 1과 0을 설정하며 학습을 통해 최적화합니다.
</details>

<img src="https://github.com/user-attachments/assets/c5c532c5-d5a8-4606-8af9-1fe51bb5080b" width=300>

## 3.3.1 수식

1. Patch Embedding & [CLS]token

$$z_0 = [x_{\text{class}}; x_1^{pE}; x_2^{pE}; \cdots; x_n^{pE}] + E_{\text{pos}}, \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}, \quad E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$$

$Where$
- $x_{\text{class}}$ : [CLS]token
- $x_1^{pE}; x_2^{pE}; \cdots; x_n^{pE}$ : 각 img Patch Embedding
- $E_{\text{pos}}$ : Position Embedding
- $D$ : Transformer hidden dim
- $P^2 \cdot C$ : 각 Patch의 크기 ($P \times P$)와 채널 수($C$)를 곱한 값
- $N$ : Patch 수
  
2. Self-Attention (MSA)

$$z_{0}^{\prime} = \text{MSA}(LN(z_{l-1})) + z_{l-1}, \quad l=1 \cdots L$$

$Where$
- $\text{MSA}$ : Multi-Head Self-Attention
- $\text{LN}$ : Layer Normalization
- $z_{l-1}$ : 이전 레이어의 출력
- $L$ : Transformer의 총 레이어 수

3. MLP

$$z_l = \text{MLP}(LN(z_{l}^{\prime})) + z_{l}^{\prime}, \quad l=1 \cdots L$$

$Where$
- $\text{MLP}$ : Mulilayer Perception (Feedforward Neural Netwokr)로 비선형을 적용해서 모델의 표현 능력을 높임
- $\text{LN}$ : Layer Normalization

4. 최종 출력

$$y=LN(z_{0}^L)$$

Transformer 최종 출력
