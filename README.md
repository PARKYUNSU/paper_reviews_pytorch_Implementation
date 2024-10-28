# DeepLabV1
---
Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
https://arxiv.org/pdf/1412.7062

저자 : Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
---



## 1. Introduction
**Deep Convolutional Neural Networks** (DCNN)은 image classification, object detection, fine-grained categorization 등 컴퓨터 비전의 여러 방면으로 시스템을 향상시켰습니다.

이러한 성공의 주된 부분은 Image의 Spatial invariance으로 인해 단계적으로 학습할 수 있다는 점으로 볼 수 있겠습니다. 

그러나 이러한 Saptial Invariace는 자세 추정 및 Segmentation 등 정확한 위치 지정이 필요한 작업에서는 불 필요할 수 있습니다.

논문에서는 2가지의 문제점을 거론합니다.

## 1.1 문제점

### **1) DCNN에서 수행되는 MaxPooling과 Stride로 인한 이미지 해상도 저하**
 
 - 특징 맵(feature map)의 해상도가 점점 줄어들면서 자세한 부분을 놓칠 수 있고, 서로 다른 크기의 객체를 인식하는 것이 어려워집니다.

### **2) DCNN의 Spatial Invariance의 문제점**
   
- 다운샘플링 과정에서 위치 정보가 불확실해집니다. 이러한 특성 때문에, 정밀한 경계선 및 세부적인 위치를 못 찾게 됩니다.

논문의 저자는 2가지의 문제점을 해결하기 위해 2가지 해결방안을 제안합니다.

## 1.2 해결방안

### **1) Atrous Convolution (Dilation convolution)**

### **2) Fully-Connected Conditional Random Field / Dense Conditional Random Field (CRF)**


## 2. Atrous Convolution (Dilation convolution)

DCNN의 첫 번째 문제인, Downsampling은 MaxPooling과 Stride로 인해 깊은 층으로 가면 갈 수록 이미지 해상도 저하 문제를 일으킵니다.

논문에서는 이 문제점을 해결하기 위해, 일부 max-pooling 계층을 제거하고 대신 **Dilation convolution**을 도입하여 해상도를 유지하면서 더 넒은 수용 영역을 확보합니다.

Dilation convolution의 설명은 다음과 같습니다.

dilation을 한 마디로 말하자면 convolution 커널의 간격을 의미합니다.

dilation이 2라면 커널 사이의 간격이 2가 되고, 커널의 크기가 (3,3)이라면 (5,5) 커널과 동일한 넓이가 됩니다.

필터 내부에 zero padding을 추가해 강제로 receptive field를 늘리게 됩니다.

즉, weight가 있는 부분을 제외한 나머지 부분은 전부 0으로 채워지게 됩니다.

기존의 receptive field의 한계를 극복하고 좀 더 넓은 간격의 커널을 적은 리소스로 사용할 때 많이 활용됩니다.

###### 케라스에서는 dilation rate로 파이토치에서는 dilation으로 입력을 받는다.

![image](https://github.com/user-attachments/assets/b1a3b425-8910-4450-94c2-2b1047746fd0)

## 2.1 Dilation convolution의 계산 방법

아래 그림에서 파란색 weight가 있는 픽셀 사이에 0이 들어간다고 생각하면 되는데, 여기서 rate parameter = 2이면 계산은 다음과 같습니다.

$k$ = 3 (일반적인 CNN 커널사이즈 3)

$r$ = 2

$k_e$ = $k$ + ($k$ - 1)($r$ - 1)
5 = 3 + (3 - 1)(2 - 1)
$k_e$ = 5 (5 X 5 커널)

![image](https://github.com/user-attachments/assets/6bcc6fd3-0774-482f-8a70-5a9176d47b0d)

![image](https://github.com/user-attachments/assets/7b0dfabe-4625-49e4-b01a-05dae4273722)


논문에서는 Dilation convolution를 Conv5와 FC6에서 rate=2, rat2 12로 지정해서 사용했다.

## 2.2 Bilinear Intrepolation

마지막은 원본 해상도로 복구하기 위해서 upsampling은 Bilinear Intrepolation을 사용했다.

Bilinear Intrepolation 선형 보간법으로 작은 이미지를 부드럽게 확장하는 효과가 있다.

![image](https://github.com/user-attachments/assets/02bd2f7f-c6cf-48f8-a684-3478153c44e5)



## 3. Fully-Connected Conditional Random Field / Dense Conditional Random Field (CRF)

두 번째 문제는, Spatial Invariance입니다. Image Classification 같은 작업에서는 Spatial Transformation 에 대해 Invariance해야합니다.

Spatial Invariance으로 인해 손실된 공간적 정보를 복원하고, 객체의 경계선을 더 정교하게 표현할 수 있습니다.

논문에서는 10번의 CRF 과정을 적용하여 세밀한 경계선을 복원하고 정확한 위치 정보를 보완 합니다.

![image](https://github.com/user-attachments/assets/e5a4e799-1f4c-4ba8-9f56-189dccbe80bd)


## 3.1 Spatial Invariance

Spatial Invariance는 어떤 이미제 물체가 나타나면 그 위치에 관계없이 감죄 된다는 의미입니다.

그로인해서, 고양이가 이미지 왼쪽 위, 오른쪽 위 등 어느 곳에 있든지 위치에 상관없이 고양이로 인식하는 것을 의미합니다.

![image](https://github.com/user-attachments/assets/de58e08a-7f32-408f-a20d-d9be5e8e2897)


## 3.2 CRF

**CRF** 는 2가지 과정으로 이루어져 있습니다.

## 3.2.1 Unray Term

  개별 픽셀의 클래스 확률(점수)을 할당 받습니다. 즉, 각 픽셀이 특정 클래스에 속할 가능성을 나타냅니다. ex) 고양이인지 배경인지 등의 확률

## 3.2.2 Pairwise Term

   픽셀간의 상호작용을 담당하는 과정으로, 근처의 픽셀들이 비슷한 레이블을 가지도록 도와줍니다.
   pairwise에서도 2가지 방식이 있습니다. 이 두 방식은 함께 적용되어 객체의 세부적인 경계를 표현하도록 도와줍니다.

   **3.2.2.1 Gaussian Pairwise Term**

   색상, 위치, 밝기 등의 외형적 유사성을 체크합니다. 두 픽셀이 색상이나 위치가 유사하면 같은 클래스로 분류되도록 유도하고, 차이가 크면 서로 다른 클래스로 나뉘도록 합니다. 이 방식을 통해 이미지 내 경계가 뚜렷한 영역을 잘 구분할 수 있도록 도와줍니다.
   
   **3.2.2.2 Smoothness Pairwise Term**

   인접한 픽셀이 같은 클래스로 분류되도록 유도합니다. 두 픽셀이 가까울수록 같은 클래스로 분류될 가능성이 높아지며, 경계선 주변의 픽셀을 부드럽게 연결하는 효과를 줍니다.
   
## 3.2.3 수식

$x$ 는 Pixels에 대한 label assignment입니다. Unray Term은 $θ_i(x_i)$ = $-logP(x_i)$를 사용하며 $P(x_i)$는 DCNN으로 계산된 개별 픽셀의 클래스 확률입니다.

$θ_{i,j}(x_i, x_j)$ = $µ(x_i, x_j)Σ^K_{m=1}w_m⋅k^m(f_i, f_j)$ 이며 $x_i$와 $x_j$이 일치하지 않을 때 $µ(x_i ,x_j)$$=$$1$ 이고 아니면 0이다(즉 Potts Model). pixel i, j가 얼마나 멀리 있든 각 pair에 대한 하나의 pairwise term이 있다. 즉, 모델의


![image](https://github.com/user-attachments/assets/e6b97a48-6ad2-49ba-b976-e20bbc563c27)




​



​







1.3 정확도 감소 문제


2. Method
논문의 저자들은 3가지 방법을 제안했습니다. 이들 각각에 대해 더 깊이 살펴보겠습니다.

2.1 밀집 특징 추출을 위한 Atrous Convolution
Atrous Convolution 방식은 해상도를 줄이지 않으면서 특징을 추출할 수 있는 방법으로, 표준 CNN의 다운샘플링으로 인한 해상도 손실 문제를 완화합니다.

이때, rate parameter를 조정하여 픽셀 간의 간격을 두고 필터를 적용함으로써 연산량을 증가시키지 않으면서 해상도를 유지하는 Dense feature extraction이 가능합니다.

이는 R-CNN의 Spatial Pyramid Pooling에서 영감을 얻은 방식으로, 각기 다른 스케일의 Atrous Convolution 계층이 병렬로 적용된 후 각각의 특징들이 결합됩니다.

2.3 정확한 경계 복구를 위한 CRF 적용
DCNN을 통해 얻게 된 특징 맵을 Bilinear Interpolation으로 원본 이미지 크기로 확대한 후, CRF를 사용해 픽셀 단위의 정밀도를 높입니다. CRF는 각 픽셀의 위치와 색상을 기준으로 주변 픽셀과의 상호작용을 통해 비슷한 컬러를 가진 인접 픽셀에 동일한 레이블을 부여함으로써 더욱 정확한 경계 복구를 가능하게 합니다.

