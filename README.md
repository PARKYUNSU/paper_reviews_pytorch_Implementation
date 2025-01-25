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


[CLS]token은 그림에서 파란색 부분으로 그 크기는 $1 \times 768$ 텐서입니다. (ViT의 hidden size) 이 [CLS]token은 Patch Embedding 맨 앞에 추가되며, 그림처럼 패치 수가 196개라면, Transformer 입력 시퀀스의 크기는 $197 \times 768$ (196개의 패치 + 1개 class token)

## 3.2.2. Positinoal Embedding

