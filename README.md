# GhostNet

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
| **Knowledge Distillation** | 큰 모델에서 작은 모델로 지식을 전달하여 학습하는 기술.        | 출력 기반 증류, 특징 기반 증류             | - 작은 모델도 높은 성능 달성
이 가능.                | - Teacher 모델 학습에 많은 자원이 필요함.       |

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

그러나, 본 논문의 저자는 위의 방법론 대신에, 기존 CNN의 Feature Map에서 생성하는 정보들을 비교하여 새로운 방법으로 경량화를 제공하고자 합니다.


## Ghost Module for More Features

<img src="https://github.com/user-attachments/assets/594f54f0-b35a-4fa7-8f0c-124bcaab7c62" width=400>
