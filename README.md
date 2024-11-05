# Inception
---

“Going deeper with convolutions”

-Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich

https://arxiv.org/pdf/1409.4842

---

## 1. Introduction

GoogLeNet 모델이 기준년(2014년) 2년 전 Krizhevsky et al에  우승한 아키텍처보다 12배 적은 파라미터를 사용하면서 정확도를 높이는 성과를 거뒀습니다. 

이 논문의 이름은 Network in Network 논문 저자인 Lin et 과 Inception 영화에서의 “We Need to go Deeper”의 밈에서 유래되었습니다. Deep이라는 단어가 두 가지 의미로 사용되는데, 하나는 Inception 모듈 이라는 a new levle of organization을 도입했다는 의미가 있고, 다른 하나는 네ㅡㅌ워크의 깊이가 증가했다라는 의미가 있습니다.

<img src="https://github.com/user-attachments/assets/53edd6f6-dfb8-4641-9769-3b2ae17ed32e" width=400>

## 2. 문제점

기존 deep neural networks의 성능을 향상시키는 가장 간단한 방법은 네트워크의 Depth 와 Width를 늘리는 것이었습니다. 많은 양의 라벨이 있는 데이터를 사용할 때, 이러한 방법을 사용하여 안전하게 훈련하는데 기여했습니다. 

그러나 여기에도 2가지의 문제점이 있습니다.

### 2.1 Overfitting 문제

네트워크의 크기가 커질수록 파라미터 수가 증가하며, 학습데이터가 제한된 경우 overfitting이 발생하기 쉽습니다. 그래서 처리가 잘된 고품질의 학습 데이터를 준비하는 데는 시간과 비용이 많이 듭니다.

### 2.2 계산 자원의 증가

네트워크의 크기가 커질수록 파라미터와 마찬가지로 계산 연산량이 기하급수적으로 증가하는 단점이 생깁니다. 그래서 효율적인 자원 분배가 가능한 모델을 만드는 이유가 여기에 있습니다.

## 3. Architecture

Inception Architecture는 CNN안에 존재하는 Optimal local sparse structure가 사용가능한 Dense Components로 표현 가능하다는 아이디어에 기초합니다.

- Optimal Local Sparse Structure
    
     네트워크가 효율적으로 정보를 처리하기 위해서는 필요한 최소한의 연결만을 사용한는 것을 의미합니다. 각 뉴런이 실제로 관련있는 뉴런과 연결되어 있으면, 즉 필요한 정보만 집중적으로 처리할 수 있으면 엄청난 네트워크가 될 수 있습니다. 그러나 이러한 구조를 구현하는 것은 비현실 적입니다.
    
- Dense Components
    
    Dense Component로 표현 가능하다는 말은, 실제로 Optimal Local Sparse Structure를 만든기보단 여러가지의 필터(Inception 에서는 1X1, 3X3, 5X5등)를 병렬로 사용해서 다양한 스케일 정보를 한 번에 포착하는 방식입니다. 이러한 방식은 희소하게 연결된 네트워크가 밀집된 구조를 통해 가까운 형태로 근사할 수 있습니다.
    
<img src="https://github.com/user-attachments/assets/7435408f-80f0-4565-a5f4-222ed520e3a9" width=600>

위의 아이디어를 근간으로 Inception Network의 주요 구조를 확인할 수 있습니다.

a)에서는 1X1, 3X3, 5X5, 3X3 max pooling을 통해서 다양한 사이즈의 receptive filed를 활용해서 학습함을 알 수 있다.

b)에서는 3X3과 5X5 Conv 앞에 1X1 Conv를 추가하여 Channel 수를 줄여서 파라미터 수를 감소시키는 구조입니다. 

다음은 1X1 Conv로 연산을 통해 파라미터구조가 어떻게 바뀌는지에 대한 예시이다.

<img src="https://github.com/user-attachments/assets/dd5d2963-fe8f-4884-b11f-1e3a65669c58" width=500>

## 4. Inception Module

Inception 모듈은 네트워크의 깊이와 너비를 효울적으로 증가시키기 위해 만들었습니다. 이 모듈은 3가지 필터를 동시에 적용하고, 그 결과를 채널 방향으로 결합합니다.

- 저수준 특징(Low-level Features)
    - 에지, 코너 등의 기본적인 패턴을 감지합니다. 작은 필터(예: 1x1, 3x3)를 사용하여 추출합니다.
- 중간 수준 특징(Mid-level Features)
    - 텍스처, 모양 등의 보다 복잡한 패턴을 감지합니다. 중간 크기의 필터(예: 3x3)를 사용하여 추출합니다.
- 고수준 특징(High-level Features)
    - 물체, 얼굴 등의 복잡한 패턴을 감지합니다. 큰 필터(예: 5x5)를 사용하여 추출합니다.
- Inception 모듈의 구성 요소
    - 1x1 합성곱 브랜치: 차원 축소 및 저수준 특징 추출.
    - 1x1 -> 3x3 합성곱 브랜치: 중간 수준의 특징 추출.
    - 1x1 -> 5x5 합성곱 브랜치: 고수준의 특징 추출.
    - 풀링 브랜치: 맥스 풀링을 통해 공간 정보를 요약하고, 1x1 합성곱으로 차원 축소.

이러한 다양한 스케일의 특징을 병렬로 추출하여 결합함으로써 모델의 표현력을 향상시킬 수 있습니다.

- 기여
    - 규모가 큰 블록과 병목을 보편화
    - 병목 계층으로 1x1 합성곱 계층 사용
    - 완전 연결 계층 대신 풀링 계층 사용
    - 중간 소실로 경사 소실 문제 해결

## 5. GoogLeNet

<img src="https://github.com/user-attachments/assets/56281947-fcc3-47c6-b33d-149f8105444a" width=700>
