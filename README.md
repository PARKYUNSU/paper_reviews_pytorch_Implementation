# FCN
 
---
## Introduction
FCN(Fully Convolutional Network)은 이미지 분류에 뛰어난 성능을 보인 CNN 모델(AlexNet, VGG16, GoogLeNet)을 Semantic Segmentation(이미지 분할) 작업에 맞게 변형한 모델입니다.

이 모델은 보통 이미지 분류를 위한 사전 학습된 네트워크를 활용하고, 이를 미세 조정하여 학습하는 전이 학습(Transfer Learning) 방식으로 구현됩니다. 이후 등장한 대부분의 이미지 분할 모델들도 FCN의 아이디어를 바탕으로 발전되었습니다.

## Semantic Segmentation Vs Instance Segmentation
### 1. 시멘틱 세그멘테이션 (Semantic Segmentation)
시멘틱 세그멘테이션은 이미지의 각 픽셀을 특정 클래스에 할당하는 작업입니다. 

즉, 동일한 클래스에 속하는 모든 픽셀들을 같은 범주로 분류합니다. 여기서 "클래스"란 고양이, 사람, 자동차와 같은 객체 유형을 의미합니다. 하지만 시멘틱 세그멘테이션은 개별 객체를 구분하지는 않습니다. 예를 들어, 이미지 내에 여러 마리의 종이 쓰레기가 있더라도 모든 종이 쓰레기는 하나의 "종이쓰레기" 클래스에 속하게 됩니다.

### 2. 인스턴스 세그멘테이션 (Instance Segmentation)
인스턴스 세그멘테이션은 시멘틱 세그멘테이션의 확장된 개념으로, **동일한 클래스 내에서 개별 객체(인스턴스)**까지 구분하는 작업입니다.

즉, 이미지 내에 여러 개의 동일 클래스 객체가 있을 경우 각각을 독립적으로 인식하고 픽셀 단위로 구분합니다. 예를 들어, 이미지 내에 2개의 종이쓰레기가 있을 때, 각각의 종이쓰레기를 독립된 개체로 구분하여 픽셀을 할당합니다.

<img src="https://github.com/user-attachments/assets/29aa9796-0717-40b0-9b1d-fdde766207e0" width="600px">


|   | **시멘틱 세그멘테이션** | **인스턴스 세그멘테이션** |
|---|------------------------|-------------------------|
| **목적** | 이미지 내 모든 픽셀을 특정 클래스에 할당 | 클래스뿐만 아니라 개별 객체까지 구분 |
| **클래스 구분** | 동일한 클래스는 하나의 그룹으로 분류 | 동일 클래스의 객체도 개별적으로 구분 |
| **복잡도** | 상대적으로 낮음 | 상대적으로 높음 |
| **예시** | 모든 종이쓰레기를 하나의 "종이쓰레기"로 처리 | 각 종이쓰레기를 별개의 개체로 처리 |

---

## FCN
<img src="https://github.com/user-attachments/assets/f8447903-0a2f-4a37-849b-f38a67f3d056" width="500px">

### 1. VGG BackBone

FCN(Fully Convolutional Network)은 VGG를 백본으로 사용하는 모델입니다. VGG는 이미지 분류에서 우수한 성능을 보여왔기 때문에, FCN은 이를 기반으로 하여 Semantic Segmentation 작업에 맞게 변형되었습니다. VGG의 컨볼루션 레이어들을 활용하여 이미지로부터 중요한 특징들을 추출하고, 그 이후에 **FC(완전연결층)**을 제거하여 1x1 컨볼루션과 업샘플링(Transpose Convolution) 과정을 통해 각 픽셀을 분류합니다. 그러나 Segmentation 작업을 위해서 변형이 필요합니다.

<img src="https://github.com/user-attachments/assets/61fafcf0-a548-49c7-a2c3-385bd293d276" width="600px">

### 2. VGG Networks Fully Connected Layer -> Convolution으로 대체

VGG 네트워크의 FC6과 FC7 층을 Convolution으로 변경합니다. 

기존의 이미지 분류에서는 위치 정보가 중요하지 않아 Fully Connected Layer를 사용했지만, Fully Connected Layer는 모든 노드를 일렬로 연결하여 위치 정보를 잃어버리는 단점이 있습니다.

이는 Segmentation 작업에 적합하지 않습니다.

아래 그림은 이 차이를 잘 보여줍니다. Convolution을 사용한 경우에는 초록색 라벨 세제의 위치 정보가 유지되지만, Fully Connected Layer를 사용할 경우 위치 정보가 손실됩니다.

<img src="https://github.com/user-attachments/assets/07ad1490-1d15-4a3a-a734-f2c7349307ff" width="500px">

<img src="https://github.com/user-attachments/assets/e41d1143-65c7-4330-acc6-14ead0f9a26f" width="500px">

### 3. Transposed Convolution을 이용해서 Input Size로 복원 및 Pixel Wise Prediction 수행

FCN은 Transposed Convolution을 이용해 원본 이미지의 크기로 복원하고, 이를 통해 각 픽셀에 대해 클래스 예측을 수행합니다.

이를 통해 공간 정보를 보존하면서도 분할 작업을 진행할 수 있습니다.

<img src="https://github.com/user-attachments/assets/23c33bd8-4edf-4c5c-896d-e194b370f9d6" width="600px">


### 4. FCN 구조

FCN은 크게 네 단계로 나눌 수 있습니다

Convolution Layer를 통해 Feature 추출: CNN의 컨볼루션 층을 통해 이미지에서 중요한 특징을 추출합니다.

1x1 Convolution으로 채널 수 조정: 1x1 컨볼루션을 사용해 추출된 피처 맵의 채널 수를 클래스 수에 맞게 조정하여 '클래스 존재 히트맵'을 생성합니다.

Up-sampling: 낮은 해상도의 히트맵을 Transposed Convolution으로 업샘플링하여 입력 이미지와 동일한 크기의 맵으로 복원합니다.

네트워크 학습: 최종 피처맵과 라벨 피처맵의 차이를 기반으로 네트워크를 학습합니다.

<img src="https://github.com/user-attachments/assets/657bf405-f76f-42af-b2d3-2a834abed5b9" width="600px">

### 5. Upsampling 문제와 해결책
VGG16 네트워크에서 입력 이미지 크기가 224x224인 경우, 5개의 컨볼루션 블록을 통과하면 피처 맵 크기는 7x7로 줄어듭니다.

이는 입력 이미지의 32x32 픽셀 영역을 하나의 피처 맵 픽셀이 대표하게 되므로, 위치 정보가 대략적으로만 유지됩니다. 

이 상태에서 Transposed Convolution을 사용해 업샘플링하면 원본 이미지 크기로 복원되지만, 세부 위치 정보가 뭉개지는 문제가 발생합니다.

이를 해결하기 위한 직관적인 방법은 Down-sampling을 없애는 것입니다. 그러나 이는 연산량을 크게 증가시키기 때문에 현실적으로는 어렵습니다.

대신 FCN은 Skip Architecture를 사용하여 이 문제를 해결합니다.

### 6. Skip Architecture
Skip Architecture는 상위 레이어에서 추출된 피처 맵을 업샘플링 과정에서 결합하여 위치 정보 손실을 줄이는 방법입니다.

이를 통해 디테일한 분할 맵을 얻을 수 있습니다. FCN은 이 구조를 활용하여 성능을 향상시켰습니다.

### 7. FCN의 확장 모델

FCN-32s: 피처 맵을 한 번에 32배 업샘플링하여 원본 크기로 복원합니다.
FCN-16s: 32배 대신, 16배 업샘플링 후 중간 피처 맵과 결합(Skip)을 합니다.
FCN-8s: 더 많은 세부 정보를 보존하기 위해 8배 업샘플링합니다.

결과적으로 FCN-32s에서 FCN-8s로 갈수록 더 정교한 위치 정보가 보존됩니다.

<img src="https://github.com/user-attachments/assets/972f2712-2536-45bf-b74c-64988a9cc911" width="600px">

### FCN의 한계

1) 큰 object의 경우 지역적인 정보만 예측
2) 같은 object의 경우 다르게 labeling
3) 작은 object가 무시되는 문제가 있음

<img src="https://github.com/user-attachments/assets/81c8aa2b-2693-4666-9979-21d387a7070f" width="500px">
