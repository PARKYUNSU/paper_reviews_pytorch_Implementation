# Xception

---

“Deep Learning with Depthwise Separable Convolutions”

-Franc¸ois Chollet

Google, Inc

https://arxiv.org/pdf/1610.02357

---

## 1. Introduction

Inception 아키텍처 및 후속작

-   Inception V1(GoogLeNet) - 2014년
-   Inception V2 - 2015년
-   Inception V3 - 2015년
-   Inception-ResNet - 2016년

Inception 아키텍처는 이전 Network-in-Network 모델에서 영감을 받았으며, 그 이후 ImageNet Dataset을 비롯하여 좋은 성능을 보여줬습니다.

Inception의 기본 아이디어는 단일 Conv 층의 중첩이 아닌, 모듈화 Convolution을 기반으로 여러 분기로 나누어 처리하는 것 입니다.

논문의 저자는 Inception의 가설을 통해서 더 강력한 Xception의 가설을 수립하여, 그 가설이 성립하도록 새로운 아키텍처를 설계했습니다.

## 2. The Inception hypothesis (Inception 가설)

---

## 2.1 Inception Hypothesis

Inception은 `교차 채널 상관 관계(Cross-Channel Correlations)`와 `공간적 상관 관계(Spatial Correlations)`를 분리하여 더욱 효율적으로 학습했다 라는 가설입니다.

> **1x1 Conv를 사용해 교차 채널 상관 관계를 학습하고, 그 후 3x3 또는 5x5 Conv로 공간적 상관 관계를 학습합니다**

<img src="https://github.com/user-attachments/assets/7ec44a9c-ce27-4ea2-82bc-bdb293180806" width=400>

## 2.1.1 교차 채널 상관 관계 (Cross-Channel Correlations)

> **1X1 컨볼루션을 통해 교차 채널 상관 관계를 학습한다.**

→ 채널 간의 다양한 조합을 통해 새로운 특징을 생성한다.

→ 여러 채널 간의 관계를 더 작은 Feature Map 공간으로 압축할 수 있어 공간적 상관 관계 학습이 더 좋아진다.

ex) RGB 3개 채널에 1X1 Conv 적용

1. RGB

    R= 100, G= 150, B= 200

2. 1X1 Conv Weight

    $W_R$ = 0.2, $W_G$ = 0.3, $W_B$ = 0.5

3. Ouput

    (100 X 0.2) + (150 X 0.3) + (200 X 0.5) = 165

    RGB 값을 결합하여 165 라는 새로운 값을 생성. 즉, 1X1 Conv를 여러 필터에 적용하면, 각 필터 마다 여러 개의 채널이 생성 됩니다.

## 2.1.2 공간적 상관 관계 (Spatial Correlations)

> 3X3 또는 5X5 Conv로 공간적 상관 관계를 학습합니다.

→ 공간적 상관 관계는 이미지 내의 위치적 관계

→ 픽셀들이 이미지 내에서 어떻게 배열되고 있는지 정보 (객체 형태, 모양, 경계 등을 인식)

→ 1x1 Conv가 먼저 정보를 요약하면, 각 채널이 특정 패턴을 강조하도록 분리가 가능, 그 다음 단계의 공간적 필터가 더 중요 정보 집중적으로 학습할 수 있게 된다. (ex) a 필터 : 가장자리 정보에 민감, b 필터 : 생상이나 질감 정보 etc)

→ 1x1 Conv로 채널수가 줄어, 3X3 또는 5X5의 입력 채널이 감소하여 파라미터가 줄어든다.

즉, Inception 모듈은 1X1 Conv 교차 채널 상관 관계를 먼저 학습하고 3X3 or 5X5 Conv로 공간적 상관 관계를 학습합니다. 이를 통해서 더 적은 파라미터로 학습할 수 있었다라는 아이디어 입니다.

## 2.2 Depthwise Separable Convolution

일반적인 Conv 연산은 다음 그림과 같이 하나의 커널로 동시에 학습하나

<img src="https://github.com/user-attachments/assets/69734cf8-b279-4e47-b569-0d66a851aecd" width=400>

Depthwise Separable Convolution 에서는 `Depth Wise(공간 축)`, `Point Wise(채널 축)` 로 2 단계로 나눕니다.

Inception 과 마찬가지로

1. **Depthwise Convolution**으로, 각 채널에 대해 독립적으로 공간적 상관 관계를 학습

1. **Pointwise Convolution** (1x1 컨볼루션)으로, 학습된 공간적 상관 관계를 새로운 채널 공간으로 매핑하여 교차 채널 상관관계를 학습

<img src="https://github.com/user-attachments/assets/70601855-f3bd-4e19-9f9e-a40c7e7df013" width=400>

그러나 Inception 모듈과 유사하지만 더 극단적으로, 교차 채널 상관 관계와 공간적 상관관계를 완전히 분리하는 형태

## 3. The Xception Architecture (Xception 가설)

기존 Convolution과 Depthwise Separable Convolutioin 사이에 다양한 중간 형태의 Inception 모듈이 존재할 수 있으며, 이러한 중간 형태의 특성은 아직 탐구되지 않았다 라는 점에서 Xception 모델이 등장했습니다.

Xception 아키텍처는 Inception 가설을 한 단계 더 확장한 것입니다.

“기존 가설이었던 교차 채널 상관 관계와 공간적 상관 관계를 분리할 수 있다” 라는 가설에서

→ “두 관계들을 동시에 학습할 필요 없이 완전히 분리하여 학습이 가능” 하다로 발전

즉, Depthwise Separable Convolution을 통해 교차 채널 상관 관계를 1X1 Conv로 공간적 상관 관계를 독립적인 Depthwise Separable Convolution 으로 처리

## 3.1 Xception Architecture

Xception 아키텍처

36개의 Depthwise Separable Conv, 14개의 Inception Module 로 구성됩니다.

Xception은 간단한 구조로, Keras나 TensorFlow-Slim을 사용해 30~40줄의 코드로 구현할 수 있으며, Inception V2나 V3보다 훨씬 정의하기 쉽습니다.

<img src="https://github.com/user-attachments/assets/331ff43c-946c-4815-b10f-341bc95d6f2b" width=400>

Figure 1. - 기존 Inception Module

<img src="https://github.com/user-attachments/assets/d88a8d1d-822c-4e34-a023-5edf500d9f50" width=400>

Figure 2. - 해당 Inception module을 1X1 Conv로 재구성하고 Output channel이 겹치지 않는 부분에 대해서 Spatial Conv(3X3)으로 재구성

<img src="https://github.com/user-attachments/assets/04dc7074-0828-4fb8-8e8a-a8bce8428ce0" width=400>

Figure 3. - Figure2, Figure3은 서로 동일한 형태로 branch1 은 Input에 대해 1X1 Conv를 수행하고 Output Channels에 3X3 Conv를 수행, branch2, 3 도 동일 과정을 거쳐 Concat

<img src="https://github.com/user-attachments/assets/28cfeb1a-4276-4037-908f-9c65c9c2d267" width=400>

Figure 4. - 가설로 이 방법으로 Cross-channel correlation과 Spatial Correlation을 완전하게 분리 학습 가능하다

<img src="https://github.com/user-attachments/assets/f5b79c79-382a-46d4-a820-25f7553708d9" width=700>

Xception Model Architecture

## 4. Experimental Evaluation

## Xception vs. Inception V3 비교

### 1. 실험 설정

-   Xception 논문에서는 Xception과 Inception V3 모델을 비교하기 위해 ImageNet과 JFT 데이터셋에서 실험을 진행하였습니다.
-   ImageNet의 1000개 클래스 단일 라벨 분류 작업이고, 두 번째는 JFT 데이터셋의 17,000개 클래스 다중 라벨 분류 작업입니다,

### 2. Xception과 Inception V3의 주요 차이점

-   **모델 구조**:
    Xception은 일반적인 Inception 모듈 대신 Depthwise separable convolution을 사용하여 효율성과 표현력을 향상시키고자 했습니다.

-   **방식**:
    -   _Inception V3_: Inception 모듈에서 일반 합성곱을 사용.
    -   _Xception_: Depthwise separable convolution을 모든 합성곱을 대체하여 계산량을 줄이면서 성능을 유지.

### 3. Experiment Result

| 데이터셋     | 모델         | Top-1 Acc (%) | Top-5 Acc (%) | Paremter (백만) | FLOPs (십억) |
| ------------ | ------------ | ------------- | ------------- | --------------- | ------------ |
| **ImageNet** | Inception V3 | 77.9          | 93.7          | 23.8            | 5.7          |
|              | Xception     | 79.0          | 94.5          | 22.8            | 8.4          |
| **JFT-300M** | Inception V3 | 78.9          | 94.0          | 23.8            | 5.7          |
|              | Xception     | 80.0          | 94.8          | 22.8            | 8.4          |

-   **Top-1 및 Top-5 Acc**: Xception은 두 데이터셋 모두에서 약간 더 높은 정확도를 달성하였으며, Depthwise separable convolution이 특징 표현 향상에 효과적임을 보여주었습니다.
-   **파라미터 및 FLOPs**: Xception은 파라미터 수가 Inception V3보다 약간 적으나, Depthwise separable convolution해 FLOPs는 더 많이 필요합니다.

### 4. Optimization Configuration

### ImageNet

-   Optimizer : SGD
-   Momentum : 0.9
-   Initial Learning Rate : 0.045
-   Learning Rate Decay : decay of rate 0.94 every 2
    epochs

### JFT-300M

-   Optimizer : RMSprop
-   Momentum : 0.9
-   Initial Learning Rate : 0.001
-   Learning Rate Decay : decay of rate 0.9 every 3,000,000 samples

### 5. Summary

1. **높은 정확도와 비슷한 복잡도**: Xception은 파라미터 수는 유사하게 유지하면서도, Inception V3보다 높은 정확도를 기록했습니다.
2. **높은 FLOPs**: 깊이별 분리 합성곱의 추가 FLOPs는 높은 정확도를 얻기 위해 필요한 Trade-off를 나타냅니다.

## 5. 타 Model과의 성능 비교 표

<img src="https://github.com/user-attachments/assets/3ba4b457-0121-4399-a853-c386bef2d20b" width=700>
