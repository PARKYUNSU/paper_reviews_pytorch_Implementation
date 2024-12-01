# GhostNet

"More Features from Cheap Operations"

-Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu

https://arxiv.org/pdf/1911.11907
___

## Introduction

딥러닝에서 CNN은 여러 ComputerVision Task에 좋은 성능을 보여줬습니다.

그러나 높은 정확도를 보여주는 CNN은 많은 파라미터와 연산(FLOPs)을 요구합니다.

예를들어 ResNet-50은 약 25.6M 파라미터와 4.1 BFLOPs가 필요합니다.

이러한 계산량은 모바일 Device에서는 비효율적입니다. 따라서 효율적이고 경량화된 네트워크 설계가 주목받고 있습니다.

---
### 1. 가중치 제거 방법

| **Technique**         | **정의**                                                    | **종류**                                 | **장점**                                             | **단점**                                         |
|------------------------|------------------------------------------------------------|------------------------------------------|-----------------------------------------------------|------------------------------------------------|
| **Pruning**            | 중요하지 않은 가중치, 뉴런, 또는 채널을 제거하는 기술.       | 가중치 Pruning, 뉴런/채널 Pruning         | - 모델 크기 감소.<br>- 계산량 감소.                  | - 과도한 Pruning은 성능 저하를 유발할 수 있음.  |
| **Low-bit Quantization** | 가중치와 활성화를 Low-bit 값으로 표현하는 기술.               | 이진, 삼진, 8비트 양자화                  | - 메모리 사용량 감소.<br>- 계산 속도 향상.             | - 정보 손실로 인해 성능이 저하될 가능성 있음.   |
| **Knowledge Distillation** | 큰 모델에서 작은 모델로 지식을 전달하여 학습하는 기술.        | 출력 기반 증류, 특징 기반 증류             | - 작은 모델도 높은 성능 달성이 가능.                | - Teacher 모델 학습에 많은 자원이 필요함.       |

---
### 2. Architecture

| **Model**             | **Key Technique**                         | **Description**                                                                 |
|-----------------------|-------------------------------------------|---------------------------------------------------------------------------------|
| **Xception**          | Depthwise Convolution                    | Depthwise convolution을 사용하여 모델 파라미터를 더 효율적으로 활용.             |
| **MobileNetV1**       | Depthwise Separable Convolution          | Depthwise separable convolution을 사용하여 경량 신경망 구현.                    |
| **MobileNetV2**       | Depthwise Separable Convolution + Inverted Residual Block | Inverted residual block 도입으로 효율성 향상.                                  |
| **MobileNetV3**       | Depthwise Separable Convolution + AutoML | AutoML 기술을 활용해 적은 FLOPs로 더 나은 성능을 달성.                          |
| **ShuffleNetV1**      | Channel Shuffle Operation                | Channel shuffle 연산으로 채널 그룹 간 정보 흐름 교환을 개선.                    |
| **ShuffleNetV2**      | Channel Shuffle Operation + Hardware-Aware Design | 실제 하드웨어에서 속도를 고려한 컴팩트 모델 설계.                             |


## Ghost Module for More Features

기존 잘 학습된 CNN 네트워크는 Feature Map에는 풍부하지만, 중복적인 정보가 존재한다고 얘기하고 있습니다.

<img src="https://github.com/user-attachments/assets/594f54f0-b35a-4fa7-8f0c-124bcaab7c62" width=400>

| ResNet-50의 첫 번째 Residual Groupd에서 생성된 Feature Map 시각화

위의 그림에서 많은 쌍이 서로 유사하며, 서로 Ghost(유령) 처럼 보입니다. 같은 색의 상자로 표시된 유사한 특징 맵 쌍의 예를 확인할 수 있습니다.

이러한 중복성은 딥러닝 네트워크의 중요한 특징일 수 있으나, 논문에서는 중복성을 피하고 *cheap operations(값싼 연산)* 으로 대체하고자 합니다.

---

### Feature Map (ResNet-50)

---

#### Input

<img src="https://github.com/user-attachments/assets/6a60586f-a758-4316-99ae-03218b28dfb0" width=300>

#### Output

<img src="https://github.com/user-attachments/assets/c727a7a5-fad2-442c-a10e-870a794ce07d" width=400>

| 전체 이미지 : [ResNet-50 Feature Map](img/ResNet-50_feature_map.png)

---

<img src="https://github.com/user-attachments/assets/660eb7f9-b274-4189-b9f8-ad286f004761" width=300>

| Architecture of ResNet-50

ResNet-50의 2번째 레이어의 Feature Map (256개인 이유)

| Layer     | Number of Feature Maps |
|-----------|-------------------------|
| Conv1     | 64                      |
| Conv2_x   | 256                     |
| Conv3_x   | 512                     |
| Conv4_x   | 1,024                    |
| Conv5_x   | 2,048                    |

---

<img src="https://github.com/user-attachments/assets/f96f326f-3179-46c1-9cf1-fee0043f3fc6" width=500>


# Ghost Module

### 1. 기본 구조

입력 데이터 $X$를 convolution 연산으로 처리하여 $Y$라는 출력 특징 맵을 생성합니다:

$$
Y = X \ast f + b
$$

여기서:

- $f$: 필터
- $\ast$: convolution 연산
- $b$: 편향(bias)

일반적으로 $f$와 $b$는 크고 복잡하여 많은 FLOPs와 메모리를 소모합니다.

---

### 2. Ghost Module의 아이디어

Ghost Module은 convolution 연산을 다음 두 단계로 분리합니다:

#### **1) Intrinsic Feature Maps 생성**
Convolution 필터 $f_0$를 사용하여 소수의 주요 특징 맵 $Y_0$를 생성합니다:

$$
Y_0 = X \ast f_0
$$

여기서:

- $f_0$: 일반 필터
- $m$: 주요 특징 맵의 총 수 ($m < n$, $n$은 전체 출력 채널 수)

#### **2) Ghost Feature Maps 생성**
$Y_0$를 기반으로 여러 저렴한 연산(예: Depthwise Convolution, 선형 변환 등)을 수행해 ghost feature maps $Y_{ghost}$를 생성합니다:

$$
y_{ij} = \Phi_{i,j}(y_{0i}), \, \forall i = 1, \ldots, m, \, j = 1, \ldots, s
$$

여기서:

- $\Phi$: 선형 연산(cheap operation)
- $s$: 각 주요 특징 맵에서 생성될 ghost feature의 수

---

### 3. 최종 출력

최종 출력 $Y$는 intrinsic feature maps $Y_0$와 ghost feature maps $Y_{ghost}$의 결합입니다:

$$
Y = [Y_0, Y_{ghost}]
$$






