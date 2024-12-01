# GhostNet

"More Features from Cheap Operations"

-Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu

https://arxiv.org/pdf/1911.11907
___

## 1. Introduction

딥러닝에서 CNN은 여러 ComputerVision Task에 좋은 성능을 보여줬습니다.

그러나 높은 정확도를 보여주는 CNN은 많은 파라미터와 연산(FLOPs)을 요구합니다.

예를들어 ResNet-50은 약 25.6M 파라미터와 4.1 BFLOPs가 필요합니다.

이러한 계산량은 모바일 Device에서는 비효율적입니다. 따라서 효율적이고 경량화된 네트워크 설계가 주목받고 있습니다.

---
### 1) 가중치 제거 방법

| **Technique**         | **Description**                                                    | **종류**                                 | **장점**                                             | **단점**                                         |
|------------------------|------------------------------------------------------------|------------------------------------------|-----------------------------------------------------|------------------------------------------------|
| **Pruning**            | 중요하지 않은 가중치, 뉴런, 또는 채널을 제거하는 기술.       | 가중치 Pruning, 뉴런/채널 Pruning         | - 모델 크기 감소.<br>- 계산량 감소.                  | - 과도한 Pruning은 성능 저하를 유발할 수 있음.  |
| **Low-bit Quantization** | 가중치와 활성화를 Low-bit 값으로 표현하는 기술.               | 이진, 삼진, 8비트 양자화                  | - 메모리 사용량 감소.<br>- 계산 속도 향상.             | - 정보 손실로 인해 성능이 저하될 가능성 있음.   |
| **Knowledge Distillation** | 큰 모델에서 작은 모델로 지식을 전달하여 학습하는 기술.        | 출력 기반 증류, 특징 기반 증류             | - 작은 모델도 높은 성능 달성이 가능.                | - Teacher 모델 학습에 많은 자원이 필요함.       |

---

### 2) Networks

| **Model**             | **Key Technique**                         | **Description**                                                                 |
|-----------------------|-------------------------------------------|---------------------------------------------------------------------------------|
| **Xception**          | Depthwise Convolution                    | Depthwise convolution을 사용하여 모델 파라미터를 더 효율적으로 활용.             |
| **MobileNetV1**       | Depthwise Separable Convolution          | Depthwise separable convolution을 사용하여 경량 신경망 구현.                    |
| **MobileNetV2**       | Depthwise Separable Convolution + Inverted Residual Block | Inverted residual block 도입으로 효율성 향상.                                  |
| **MobileNetV3**       | Depthwise Separable Convolution + AutoML | AutoML 기술을 활용해 적은 FLOPs로 더 나은 성능을 달성.                          |
| **ShuffleNetV1**      | Channel Shuffle Operation                | Channel shuffle 연산으로 채널 그룹 간 정보 흐름 교환을 개선.                    |
| **ShuffleNetV2**      | Channel Shuffle Operation + Hardware-Aware Design | 실제 하드웨어에서 속도를 고려한 컴팩트 모델 설계.                             |


## 2. Ghost Module for More Features

기존 잘 학습된 CNN 네트워크는 Feature Map에는 풍부하지만, 중복적인 정보가 존재한다고 얘기하고 있습니다.

<img src="https://github.com/user-attachments/assets/594f54f0-b35a-4fa7-8f0c-124bcaab7c62" width=400>

| ResNet-50의 첫 번째 Residual Groupd에서 생성된 Feature Map 시각화

위의 그림에서 많은 쌍이 서로 유사하며, 서로 Ghost(유령) 처럼 보입니다. 같은 색의 상자로 표시된 유사한 특징 맵 쌍의 예를 확인할 수 있습니다.

이러한 중복성은 딥러닝 네트워크의 중요한 특징일 수 있으나, 논문에서는 중복성을 피하고 *cheap operations(값싼 연산)* 으로 대체하고자 합니다.


### 2.1 Feature Map (ResNet-50)

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

## 3. Ghost Module

<img src="https://github.com/user-attachments/assets/f96f326f-3179-46c1-9cf1-fee0043f3fc6" width=500>

### 1) 기본 구조

입력 데이터 $X$를 convolution 연산으로 처리하여 $Y$라는 출력 특징 맵을 생성합니다:

| (a) The convolution layer.

$$
Y = X \ast f + b
$$

$Where$

- $f$: 필터
- $\ast$: convolution 연산
- $b$: 편향(bias)

이때, FLOPs는 $n⋅h^′⋅w^′⋅c⋅k⋅k$으로 그 수가 커질수 밖에 없다.

---

### 2) Ghost Module

Ghost Module은 convolution 연산을 다음 두 단계로 분리:

#### **1) Intrinsic Feature Maps**

Convolution 필터 $f^{'}$ 를 사용하여 소수의 주요 특징 맵 $Y^{'}$ 를 생성합니다:

$$
Y^{'} = X \ast f^{'}
$$

$Where$

- $Y^{'}$ : $m$개의 intrinsic Feature Map
- $f^{'}$: 일반 필터
- $m$: 주요 특징 맵의 총 수 ($m < n$, $n$은 전체 출력 채널 수)

#### **2) Ghost Feature Maps**

$Y^{'}$를 기반으로 여러 Cheap linear 연산(Depthwise Convolution, 선형 변환 등)을 수행해 ghost feature maps $Y_{ghost}$를 생성합니다:

$$
y_{ij} = \Phi_{i,j}(y^{'}_{i})$$ $$\forall i = 1\ldots, m$$  $$j = 1, \ldots, s$$


$Where$

- $y^{'}_{i}$: $i$번째 Intrinsic feature map
- $\Phi$: 선형 연산(cheap operation)
- $s$: 각 주요 특징 맵에서 생성될 ghost feature의 수

### 3) Final

최종 출력 $Y$는 intrinsic feature maps $Y^{'}$와 ghost feature maps $Y_{ghost}$의 결합입니다:

$$
Y = [Y^{'}, Y_{ghost}]
$$

---

### 4) Speed-Up 비율 계산

Ghost Module에서 **linear operations**와 **identity mapping**을 speed-up 비율은 아래와 같이 계산됩니다:

#### Speed-Up 비율 공식:

$$
r_s = \frac{n \cdot h' \cdot w' \cdot c \cdot k \cdot k}{\frac{n}{s} \cdot h' \cdot w' \cdot c \cdot k \cdot k + (s - 1) \cdot \frac{n}{s} \cdot h' \cdot w' \cdot d \cdot d}
$$

이를 간단히 정리하면:

$$
r_s \approx \frac{c \cdot k \cdot k}{\frac{1}{s} \cdot c \cdot k \cdot k + \frac{(s - 1)}{s} \cdot d \cdot d}
$$

$$
r_s \approx \frac{s \cdot c}{s + c - 1} \approx s
$$

### 설명

1. **$n$: 출력 채널 수**, **$s$: Ghost factor**, **$c$: 입력 채널 수**
2. **$k \cdot k$**: convolution 연산의 커널 크기, **$d \cdot d$**: linear operation에서 사용하는 커널 크기
3. Ghost Module은 intrinsic feature maps와 cheap operations로 구성되므로, 연산의 비효율성을 줄일 수 있음.
4. Ghost factor $s$가 클수록 계산 효율성이 더 높아짐.

---

## 4. Ghost Bottleneck
Ghost Bottleneck은 다음과 같이 구성

### 4.1 Ghost Module 1
 
  #### 4.1.1 Depthwise Convolution

   - 공간 정보를 처리하며, stride에 따라 특징 맵의 크기를 줄이거나 유지합니다.

   - stride=2일 경우, 특징 맵의 크기를 절반으로 다운샘플링합니다.

   - stride=1일 경우, 특징 맵의 크기를 유지합니다.

| **속성**              | **Stride=1**                          | **Stride=2**                                 |
|-----------------------|---------------------------------------|---------------------------------------------|
| **특징 맵 크기 변화**   | 동일 (크기 유지)                      | 크기 절반으로 다운샘플링                     |
| **Depthwise Convolution** | 공간 정보 유지 (`stride=1`)             | 공간 정보 다운샘플링 (`stride=2`)             |
| **Shortcut 연결**      | 직접 연결 (Identity Connection)        | 1×1 Convolution으로 크기 맞춤               |
| **적용 목적**          | 세밀한 공간 정보 유지                  | 연산량 감소 및 추상적 특징 학습              |
| **사용 위치**          | 네트워크 중간 레이어                   | 네트워크의 다운샘플링 전환 계층              |


  #### 4.1.2 Squeeze-and-Excitation(SE) Module

  - 입력 특징 맵의 채널별 중요도를 조정

    | [SE-Net 내용](https://github.com/PARKYUNSU/SE-Net)


### 4.2 Ghost Module 2

  #### 4.2.1 Shortcut Connection:

   - 입력과 출력이 동일한 크기일 때, 입력 특징 맵을 직접 연결(stride=1)

   - 입력 크기와 출력 크기가 다르면(if stride=2), 1×1 convolution으로 크기를 맞춘 후 연결

<img src="https://github.com/user-attachments/assets/162de10c-83b9-44a2-82d9-19b37d05d4fc" width=400>

| Ghost Bottle Neck (left: stride=1, Right: stride=2)

## 5. Architecture

<img src="https://github.com/user-attachments/assets/503066ef-eeb8-4db1-9407-d739dc81bd14" width=600>

---
## 6. Result


---
