# DeConvNet
 
---

“Learning Deconvolution Network for Semantic Segmentation”

— Hyeonwoo Noh, Seunghoon Hong, Bohyung Han

Department of Computer Science and Engineering, POSTECH, Korea

https://arxiv.org/pdf/1505.04366

---

## 1. Introduction

기존의 Classification을 위한 CNN 아키텍처를 Fully Convolution Network로 변환한 FCN은 이미지에서 지역적인 부분을 분류를 수행하고, 픽셀 단위 라벨링을 위해 간단한 Transposed Convolution(FCN : Bilinear Interpolation)을 수행합니다. 추가적으로 정밀한 분할을 위해 필요시 CRF(Conditional Random Field)를 적용합니다.

하지만 FCN 기반 Segmentation 작업에 2가지 한계점이 있습니다.

1. **Fixed-size receptive field**
    
    FCN은 고정된 receptive field로 인해 오직 하나의 스케일 이미지만 처리할 수 있어, 객체 크기에 따라 오분류가 발생할 수 있습니다.
    
    큰 객체는 여러 개의 작은 객체로 쪼개지고, 작은 객체는 무시되거나 배경으로 간주됩니다.
    
    예시: 아래 그림 (a)에서 큰 객체가 작은 객체로 분해됨.
    
    아래 그림 (b)에서 작은 객체가 거의 검출되지 않음.
    
    그래서 FCN은 skip architecture를 통해 보완하려 했으나 근본적인 해결은 어려웠습니다.
    
2. **간단한 Deconvolution 과정**
    
    FCN은 업샘플링에 bilinear interpolation을 사용하나, 이 방법은 성능 향상에 한계가 있습니다.
    
    후처리인 CRF를 사용해 성능을 개선하려 하지만, 논문에서도 bilinear interpolation을 통한 업샘플링의 한계를 지적했습니다.


<img src="https://github.com/user-attachments/assets/4bdce991-2b30-4f23-9116-bb2bad3d3799" width="600">

논문에서는 FCN의 한계점을 극복하기 위해 다음의 해결방안을 제안합니다.

1. Multi-Layer Deconvolution Network 
    
     deconvolution, unpooling, ReLU 층으로 구성된 네트워크를 학습합니다. 의미론적 분할 결과를 생성하여 FCN의 방법에서 발생하는 스케일 문제를 해결하고 객체의 세부 사항을 더욱 잘 식별합니다.
    
2. Instance-Wise Segmentation
    
    Trained Network를 객체 제안을 적용해서 Instance-wise Segmentation 결과를 얻고, 이를 결합하여  최종 Semantic Segmentation 결과를 생성합니다. 이 방법은 FCN 기반 방법에서 발생하는 스케일 문제를 해결하고, 객체의 세부 사항을 더욱 잘 식별합니다.
    

## 2. Architecture

<img src="https://github.com/user-attachments/assets/e22bd454-e6b7-4fc9-9a07-caaac32e47d1" width="800">

DeConvNet 모델은 두 부분으로 구성됩니다.

1. Convolution Network
    
    합성곱 네트워크는 입력 이미지를 다차원 특징 표현으로 변환하는 역할
    
    convolution은 VGG16에서 마지막 classification layer를 제거하여 사용합니다. 그리고 class-specific projection을 위해 2개의 fully connected layer을 추가합니다
    
2. Deconvolution Network
    
    Segmenation을 생성하는 Shape Generator 역할
    
3. Final Output
    
    네트워크의 최종 출력은 Input Image와  동일한 크기의  probability map으로, 각 픽셀이 미리 정의된 클래스 중 어느 클래스에 속할 확률을 나타냄
    

## 3. Deconvolution Network

Deconvolution Network는 Unpooling과 Deconvolution 으로 연산이 이루어집니다.

### 3.1 Unpooling

CNN 풀링 연산은 하위 층의 노이즈 활성화를 걸러내고, 영역 내에서 단일 대표 값으로 추상화합니다. 이 과정을 통해 상위 층에서 강력한 활성화만 유지해서 분류에는 도움이 되지만, 공간정보가 손실되어 의미론적 분할에 필요한 정확한 위치 정보가 부족하게 됩니다. 

이를 해결하기 위해서 Deconvolution Network에서 Unpoolin 층을 사용하여 풀리의 반대 작업을 수행하여, 활성화의 원래 크기로 복원합니다. 

풀링 과정에서 선택된 최대 활성화 위치를 switch 변수에 저장하고, 나중에 Unpooling을 할때 원래 위치로 되돌립니다.

<img src="https://github.com/user-attachments/assets/eaac4a04-0c68-4025-b382-339a6971bbef" width="500">

<img src="https://github.com/user-attachments/assets/0e783889-be45-4d64-95a0-3ba2bf757fa5" width="700">

### 3.2 Deconvolution

Unpooling으로 출력은 확대되었지만, 여전히  활성화 맵은 희소한 상태입니다 즉, sparse한 값을 갖습니다. 그래서 Deconvolution은 여러개의 학습된 필터를 사용하여 Dense한 Feature map을 만듭니다. 

Deconvolution 층도 합성곱 네트워크와 마찬가지로 객체의 다양한 형태의 세부 정보를 제공합니다. 하위 층의 필터는 객체의 전체 형태를 포착하는 반면, 상위 층의 필터는 클래스별 세부 사항을 인코딩합니다. 

아래 그림은 Deconvolution 연산과정을 보여줍니다. 2 X 2입력값을 3 X 3 영역에 배치를 하는데, 처음 입력값인 빨간색을 3 X 3 영역에 배치하고, 다음 입력 값인 파란색을 그 다음 3 X 3 영역에 배치합니다. 두 3 X 3 영역에서 겹치는 부분은 Summation 하여 Deconvolution 연산을 수행합니다.


<img src="https://github.com/user-attachments/assets/6c758579-462d-4ff9-9f99-335e387a1ecb" width="400">

다음은 Deconvolution 네트워크의  활성화 Map의 시각화에 대한 설명입니다.

Deconvolution 층을 통해서 특징이 전달되면서 객체의 세부 정보가 드러납니다. 배경은 전파 과정에서 억제가되고, 클래스와 밀접하게 강화됩니다.  상위 Deconvolution 층의 학습된 필터가 클래스별 형태 정보를 포착하는 경향을 보입니다.

<img src="https://github.com/user-attachments/assets/71f3e682-452a-4e30-b5eb-f7afe85ec362" width="800">

1. input image
2. 14 X 14 Deconv
3. 28 X 28 Deconv
4. 56 X 56 Deconv
5. 56 X 56 Unpool
6. 56 X 56 Deconv
7. 112 X 112 Unpool
8. 112 X 112 Deconv
9. 224 X 224 Unpool
10. 224 X 224 Deconv

## 4. Architecture

<img src="https://github.com/user-attachments/assets/d55fb2e0-e54b-4760-8467-1e5aa17c8feb" width="700">

## 5. Evaluation results on PASCAL VOC 2012

<img src="https://github.com/user-attachments/assets/686c5907-7a05-44ce-9b26-99b705e87394" width="800">

## 6. Result

<img src="https://github.com/user-attachments/assets/358ae2b1-002e-4474-a692-b241c4f95253" width="800">

<img src="https://github.com/user-attachments/assets/75e08160-8ab6-4be8-9a98-cd0c7f1ed780" width="400">
