# DeeplabV3_Plus
 
---

“Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation” - 2018

-Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and
Hartwig Adam

[Read the Paper](https://arxiv.org/pdf/1802.02611)

---
## 논문 리뷰 영상

[![YouTube Video](https://img.youtube.com/vi/zSiKMSNBJu0/0.jpg)](https://www.youtube.com/watch?v=zSiKMSNBJu0)

---

## 1. Introduction

DeepLabV3에서는 2가지 유형의 신경 구조를 메인으로 다룹니다.

### 1. Atrous Spatial Pyramid Pooling module
    
   여러 비율의 ASPP을 사용하여 다양한 해상도의 Feature를 얻음.
    
### 2. Encoder-Decoder Structure
    
   Encoder는 변형된 Xception 모델을 사용
    
   Decoder를 통해서 세밀한 객체의 경계를 복원
    

결과적으로, PASCAL VOC 2012와 Cityscapes 데이터에서 후처리 없이 각각 mIou를 89.0% 와 82.1% 테스트 성능을 기록했습니다.

<img src="https://github.com/user-attachments/assets/5404b727-18ea-4bc9-8b71-41fcc5723a83" width=600>

## 2. Encoder-Decoder with Atrous Convolution

### 2.1 Atrous Convolution

DeepLabV3+ 에서도 기존 DeepLabV1에서 도입된 Dilation Conv를 도입하여 해상도를 유지하면서 더 넓은 수용 영역을 확합니다.

<img src="https://github.com/user-attachments/assets/6671dfb8-e53f-4097-99ed-7373412f6c37" width=600>

---
### 2.1.1 Dilation Ratio 커널 확장 계산법 (연두색)

$k_e = k + (k-1)(r-1)$

$Where$

$k_e$ : 최종 커널 사이즈

$k$ : 커널 사이즈

$r$ : Dilation Ratio

---

$k=3$ (일반적인 CNN 커널사이즈 : 3)

$r=2$

$k_e = k + (k-1)(r-1)$

$=3+(3-1)(2-1) = 3+2 = 5$ (5 X 5 커널)

---

### 2.1.2 Dilation Ratio 가중치 거리 계산법 (빨간색)

$y[i]$ $=$ $\displaystyle\sum_{k=1}^K x[i + r ⋅ k]⋅w[k]$

$Where$

입력신호 $x$ : [2, 3, 1, 4, 5, 6, 2, 1]

가중치 $w$ : [1, -1] (길이가 2인 필터)

필터길이 $K$ : 2

Dilation Rate $r$ : 2 ($r$=1 이면 일반적인 Conv)

---

- 첫 번째 위치($i$ = 0) 에서 계산

$y[0] = x[0+2⋅1]⋅w[1] + x[0+2⋅2]⋅w[2]$

$= x[2]⋅w[1]+x[4]⋅w[2]$

$=1⋅1+5⋅(-1)= 1-5=-4$

따라서 $y[0]=-4$

- 두 번째 위치($i$ = 1) 에서 계산

$y[1]=x[1+2⋅1]⋅w[1]+x[1+2⋅2]⋅w[2]$

$=x[3]⋅w[1]+x[5]⋅w[2]$

$=4⋅1+6⋅(-1)=4-6=-2$

따라서 $y[1]=-2$

$r=2$로 첫 번째 위치와 두 번째 위치가 $r$만큼 차이가 나는 것을 확인.

---

### 2.2 Depthwise Separable Convolution

<img src="https://github.com/user-attachments/assets/0712fa5d-027c-468f-9610-309cc2367044" width=500>

1. **Depthwise Conv** : 각 채널에 대해 독립적으로 **Spatial Correlations**을 학습
2. **Pointwise Conv** : 1 x 1 Conv로 학습된 **Spatial Correlations**을 새로운 Channel 공간으로 매핑하여 **Cross-Channel Correlations**를 학습
3. **Atrous Depthwise Conv** : 기존 **Depthwise Conv**를 Dilation rate 만큼 커널 사이즈를 확장시켜 학습

### 2.3 DeepLabV3 Encoder (Aligned Xception)

DeepLabV3+ 는 **Inception Module**을 활용한 **Xception**을 변형한 Aligned Xception Backbone을 사용하게 되며 Convolution 과정을 **Atrous Conv**를 활용하여 학습을 합니다.

Inception 및 Xception 설명

[Inception](https://github.com/PARKYUNSU/pytorch_imple/tree/main/CV/Classification/Inception)

[Xception](https://github.com/PARKYUNSU/pytorch_imple/tree/main/CV/Classification/Xception)

Xception에서 변경점은 다음과 같습니다.

- Middle Flow 16번 → 더 깊게 학습
- Max Pooling 연산을 Depthwise Separable Conv로 대체
- MobileNet과 비슷한 구조화 목적으로 Depthwise Separable Conv 뒤에 Batch Normalization 과 ReLU 수행

<img src="https://github.com/user-attachments/assets/7df11124-49e1-4d63-b217-64ddf27d9c37" width=600>

### 2.4 Atrous Spatial Pyramid Pooling (ASPP)

DeepLabV3는 **Atrous Spatial Pyramid Pooling(ASPP)** 모듈을 통해서 여러 스케일에서 Atrous Conv를 적용하여 다양한 스케일에서 문맥 정보를 얻습니다.

<img src="https://github.com/user-attachments/assets/bb6ff507-3f27-4bde-b9d0-316b31ef705b" width=700>

### **Output Stride**

- **Output stride**란, 입력 이미지의 해상도에 비해 Output feature map의 해상도가 몇 배 큰지 비율
- 예를 들어, 입력 이미지 크기 (1024, 2048)이고 Feature map 크기가 (64, 128)면, output stride는 16

기존 Conv는 Pooling 및 Stride로 인해 Feature map의 크기가 작아져 더 깊이 들어가면 갈 수록 공간적 정보가 손실 되어 Segmentation에 불리할 수 있습니다.

그로 인해 DeepLabV2에서 **“Atrous Spatial Pyramid Pooling”** 을 제안함으로 Output stride를 유지하면서 계산량 및 파라미터 수를 늘리지 않고 더 큰 FOV(Filed of View)를 사용합니다.

`(batch_size, channels, height, width)`

If Output_stride = 16

- **`x1`** :  $r=1$
    - **ASPP conv1 output** (`torch.Size([3, 128, 32, 32])`):
- **`x2`** : $r=6$
    - **ASPP conv2 output** (`torch.Size([3, 128, 32, 32])`):
- **`x3`** : $r=12$
    - **ASPP conv3 output** (`torch.Size([3, 128, 32, 32])`):
- **`x4`** : $r=18$
    - **ASPP conv4 output** (`torch.Size([3, 128, 32, 32])`):
- **`x5` :** $AVG$
    - **ASPPPooling output** (`torch.Size([3, 128, 1, 1])`):
    
    최종적으로 `F.interpolate`를 통해 `32x32` 해상도로 업샘플링하여 다른 텐서들과 동일한 해상도를 가지게 만듬
    

## 2.1.5 Decoder

<img src="https://github.com/user-attachments/assets/ce58d2e1-43bf-4362-aa3c-b9695a6aa556" width=600>

논문에서는 Decoder 구조를 여러가지 방법으로 실험해서 어떤 구조가 좋은 성능을 내는지 소개합니다

1. Encoder 에서 뽑은 Low-Level Feature map의 channel을 몇으로 줄일 때 성능이 좋은지
2. 3X3 Conv 구조를 어떻게 해야 좋은지

---

### 1. Encoder 에서 뽑은 Low-Level Feature map의 channel을 몇으로 줄일 때 성능이 좋은지

Low-Level Feature map을 48로 줄일 때 성능이 제일 좋다

<img src="https://github.com/user-attachments/assets/1f87c6f1-8a0b-450e-bc49-d0a119f9350b" width=700>

### 2. 3X3 Conv 구조를 어떻게 해야 좋은지

Encoder의 Conv2를 Concat한 것을 [3X3, 256] 을 2번 했을 때가 제일 성능이 좋음

<img src="https://github.com/user-attachments/assets/2e751f78-e6ba-4f47-baa6-86364c57291b" width=700>


### Decoder Upsampling 과정

<img src="https://github.com/user-attachments/assets/8d85ac84-6317-41f3-a15c-9262d6222ec5" width=700>


## Result

<img src="https://github.com/user-attachments/assets/0360c74a-68ff-4b79-8176-8c1d44e08a4f" width=800>

### DeepLabV3+ ResNet Result (10 Epochs)
<img src="https://github.com/user-attachments/assets/1ff004fa-6f3e-43ed-bf36-9291f13692e0" width=700>


### DeepLabV3+ Xception Result (10 Epochs)

<img src="https://github.com/user-attachments/assets/268f7aa4-b6a2-4470-ad4a-ad9b4c8ea393" width=700>
