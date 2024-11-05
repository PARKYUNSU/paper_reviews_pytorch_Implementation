# FCN

"Fully Convolutional Networks for Semantic Segmentation"

-Jonathan Long, Evan Shelhamer, Trevor Darrell

https://arxiv.org/pdf/1411.4038

---
## Introduction
FCN(Fully Convolutional Network)은 이미지 분류에 뛰어난 성능을 보인 CNN 모델(AlexNet, VGG16, GoogLeNet)을 Semantic Segmentation(이미지 분할) 작업에 맞게 변형한 모델입니다.

이 모델은 보통 이미지 분류를 위한 사전 학습된 네트워크를 활용하고, 이를 미세 조정하여 학습하는 전이 학습(Transfer Learning) 방식으로 구현됩니다. 이후 등장한 대부분의 이미지 분할 모델들도 FCN의 아이디어를 바탕으로 발전되었습니다.

## Semantic Segmentation Vs Instance Segmentation
### 1. 시멘틱 세그멘테이션 (Semantic Segmentation)
시멘틱 세그멘테이션은 이미지의 각 픽셀을 특정 클래스에 할당하는 작업입니다. 

즉, 동일한 클래스에 속하는 모든 픽셀들을 같은 범주로 분류합니다. 여기서 "클래스"란 고양이, 사람, 자동차와 같은 객체 유형을 의미합니다. 하지만 시멘틱 세그멘테이션은 개별 객체를 구분하지는 않습니다. 예를 들어, 이미지 내에 여러 마리의 종이 쓰레기가 있더라도 모든 종이 쓰레기는 하나의 "종이쓰레기" 클래스에 속하게 됩니다.

### 2. 인스턴스 세그멘테이션 (Instance Segmentation)
인스턴스 세그멘테이션은 시멘틱 세그멘테이션의 확장된 개념으로, **동일한 클래스 내에서 개별 객체(인스턴스)** 까지 구분하는 작업입니다.

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

FCN-8은 피처 맵 결합을 통해 세밀한 정보가 포함되므로 복잡한 경계를 더 정확하게 추정할 수 있습니다. 반면, FCN-32는 단일 피처 맵을 32배 업샘플링하기 때문에 세밀한 경계 정보가 부족하여 대략적인 형태만 추정합니다.

<img src="https://github.com/user-attachments/assets/972f2712-2536-45bf-b74c-64988a9cc911" width="600px">

- **연산 비용과 파라미터 수**: VGG 백본을 공유하므로 세 모델 간 파라미터 수의 차이는 크지 않습니다. 하지만 **FCN-8**은 추가 피처 맵 결합으로 연산 비용이 다소 증가해 가장 느리고, **FCN-32**는 연산이 단순해 가장 빠릅니다.

- **의견**: FCN-8은 복잡한 경계와 작은 객체를 더 잘 인식할 수 있으며, 고해상도 세그멘테이션이 필요한 응용에 적합합니다. 반면, FCN-32는 비교적 빠른 연산을 요구하는 상황에서 큰 영역의 대략적인 세그멘테이션을 수행하기에 적합합니다.

### 9. FCN의 한계

1. Fixed-size receptive field

FCN은 고정된 receptive field로 인해 오직 하나의 스케일 이미지만 처리할 수 있어, 객체 크기에 따라 오분류가 발생할 수 있습니다.

큰 객체는 여러 개의 작은 객체로 쪼개지고, 작은 객체는 무시되거나 배경으로 간주됩니다.

예시:
아래 그림 (a)에서 큰 객체가 작은 객체로 분해됨.

아래 그림 (b)에서 작은 객체가 거의 검출되지 않음.

그래서 FCN은 skip architecture를 통해 보완하려 했으나 근본적인 해결은 어려웠습니다.

2. 간단한 Deconvolution 과정

FCN은 업샘플링에 bilinear interpolation을 사용하나, 이 방법은 성능 향상에 한계가 있습니다.

후처리인 CRF를 사용해 성능을 개선하려 하지만, 논문에서도 bilinear interpolation을 통한 업샘플링의 한계를 지적했습니다.


<img src="https://github.com/user-attachments/assets/81c8aa2b-2693-4666-9979-21d387a7070f" width="500px">

---

### 10. Experiment

#### 1. Experiment 1

- Dataset : CamVid (Cambridge-Driving Labeled Video Database)
  https://www.kaggle.com/datasets/carlolepelaars/camvid
- Model : FCN 8s
- Optimizer : Adam
- Num Classes : 32
- Num epochs : 100
- Lerarning rate =:0.001

### 11. Result

#### Model Performance Metrics

| Metric           | Value   |
|------------------|---------|
| Pixel Accuracy   | 84.73%  |
| Overall Accuracy | 84.73%  |
| Mean IoU         | 0.3689  |
| Precision        | 0.8465  |
| Recall           | 0.8473  |
| F1-Score         | 0.8420  |

#### Segmentation Result

<img src="https://github.com/user-attachments/assets/93d6c056-4504-4eed-b928-6b1bf771b082" width="800">

#### Loss & Accuracy

<img src="https://github.com/user-attachments/assets/08763fbb-4c3b-4431-b6f0-68a0b7b5d4b2" width="800">

---

#### Experiment 2
- Dataset : VOC 2012
- Model : FCN 8s
- Num Classes : 21
- Num epochs : 100
- Criterion : Cross Entropy Loss
- Optimizer : Adam
- Lerarning rate : 0.001

#### Segmentaion Result

<img src="https://github.com/user-attachments/assets/cb77cabf-c50e-4587-9498-6b32a055c60d" width="800">

<img src="https://github.com/user-attachments/assets/c661be13-94fe-4f86-bda7-30650dc63eec" width="800">

<img src="https://github.com/user-attachments/assets/0229be9a-f4fe-485e-8f18-f7abf4d8dc45" width="800">

<img src="https://github.com/user-attachments/assets/0c68d4d5-a55d-47db-9cc4-6ac635bdcb16" width="800">

<img src="https://github.com/user-attachments/assets/d29d3132-40ba-49a2-985d-2ea41c9eb6da" width="800">

<img src="https://github.com/user-attachments/assets/e2173684-639e-41a9-8d02-b4f56edca21e" width="800">
