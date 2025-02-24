## MobileNet V1

Link to Paper:

**“Efficient Convolutional Neural Networks for Mobile Vision Applications” -** 

— Andrew G. Howard Weijun Wang, Menglong Zhu Tobias Weyand, Bo Chen Marco Andreetto, Dmitry Kalenichenko Hartwig Adam

https://arxiv.org/pdf/1704.04861

---

Table of Contents

1. Introduction
2. Depthwise Separable Convolutions
3. Width Multiplier(*α*)
4. Resolution Multiplier(*ρ*)
5. Code
6. Experiment

---

## 1. Introduction

모바일 및 임베디드 비전 어플리케이션을 위한 효율적인 모델 MobileNet

<img src="https://github.com/user-attachments/assets/ea7b2d84-716f-41d4-a683-2ddfee02709b" width="700" height="400">


이 모델은 Object Detection, Face Attributes, Finegrain Classification, Landmark Recognition 사용에 효과적임을 보여줌

MobileNet 논문에서 제공한 3가지 기법

**1) Depthwise Separable Convolutions (깊이 분리 합성곱)**

**2) Width Multiplier(*α*)**

**3) Resolution Multiplier(*ρ*)**

## 2. Depthwise Separable Convolutions

Depthwise Separable Convolutions 기법은 합성곱을 깊이 별 합성곱과 1X1 합성곱으로 나누는 형태의 Factorized Convolution

즉, 합성곱의 연산과정을 2단계로 분해하는 방법

이 과정으로 연산 비용을 감소효과를 가져옴 

<img src="https://github.com/user-attachments/assets/3df2d826-1ad2-4d28-a4b6-ab271a418f5a" width="300" height="300">

fig.1

fig.1을 기준으로 `in_feature` 의 shape는 $M , D_F, D_F$


`Conv Layer`의 커널 사이즈 $D_K, D_K$, $N$ 개 로 정의 했을 때, 그 수식은

$D_K · D_K · M · N · D_F · D_F$

논문에서는, 기존 연산 과정을 Depth wise(공간 축), Point wise(채널 축) 으로 나눠서 계산

<img src="https://github.com/user-attachments/assets/a43de630-1496-4c29-94d4-5e5d43234638" width="500" height="650">

fig.2

fig.2 처럼 기존 연산 과정을 나눠서 계산

Input → Separate → Depth Wise Conv → Concat → Point wise Conv → Output

의 형태로 계산

Depth Wise Conv(공간 축) 는 $D_K · D_K · M · D_F · D_F$

Cost를 계산할 수 있으며,

Point wise Conv(채널 축)는 $M · N · D_F · D_F$ 계산된다.

결과적으로 기존 Conv 계산 방식에 비해 Depth Wise Conv + Point Wise Conv의 계산 방식은 다음과 같이 계산량이 감소 하는데,

$$
\frac {D_K · D_K · M · D_F · D_F + M · N · D_F · D_F}
{D_K · D_K · M · N · D_F · D_F}
$$

$$
= \frac1 N + \frac 1 {D^2_k}
$$

이러한 감소량을 보인다.

예시로,

32 X 32 X 3 ( 이미지 크기: 32X32, 채널 수: 3)

3 X 3 (커널 크기)

64 (출력 채널 수)

계산량 = $D_K · D_K · M · N · D_F · D_F$

$=32×32×3×64×3×3=1,769,472$

Depthwise Separable Convolution

 $D_K · D_K · M · D_F · D_F + M · N · D_F · D_F$

$=32×32×3×3×3+3×64×3×3=29,376$

결과적으로 계산량은  다음과 같다,

$=\frac{29,376} {1,769,472}
≈0.0167$

$\frac1 N + \frac 1 {D^2_k}
=\frac1 {64} + \frac1 {32^2} ≈0.0167$

## **3. Width Multiplier(*α*)**

MobileNet의 모델의 ‘채널 수’를 조정하여 네트워크 크기와 계산 비용을 조절한다.

네트워크의 `Width`에 해당하는 채널 개수가 조절 되어서, 모델의 경량화 효과를 줄 수 있다.

논문에서는 ***α*** 의 범위를 1-0 사이 값으로 정했으며, 실험으로는 1, 0.75, 0.5, 0.25로 사용 되었다.

연산 Cost 는 다음과 같이 조절된다.

 $D_K · D_K · aM · D_F · D_F + aM · aN · D_F · D_F$

## **4. Resolution Multiplier(*ρ*)**

MobileNet의 모델의 ‘이미지 크기’를 조절하여 네트워크 크기와 계산 비용을 조절한다.

논문에서는 ***ρ*** 의 범위를 1-0 사이 값으로 정했으며, 계산으로 Input 이미지를 224, 192, 160, 128로 조정하여 사용 되었다.

연산 Cost는 다음과 같이 조절된다.

 $D_K · D_K · aM · pD_F · pD_F + aM · aN · pD_F · pD_F$

## 6. Code

```python
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.conv1(x)
    
class DwSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = BasicConv(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = BasicConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

```python
class MobileNetV1(nn.Module):
    def __init__(self, alpha, num_classes=10, init_weight=True):
        super().__init__()

        self.alpha = alpha
        # 32*32 CIFAR IMG
        self.stem = BasicConv(3, int(32*self.alpha), kernel_size=3, stride=1, padding=1)
        # 224*224 IMG
        #self.stem = BasicConv(3, int(32*self.alpha), kernel_size=3, stride=2, padding=1)

        self.model = nn.Sequential(
            DwSepConv(int(32*self.alpha), int(64*self.alpha)),
            DwSepConv(int(64*self.alpha), int(128*self.alpha),stride=2),
            DwSepConv(int(128*self.alpha), int(128*self.alpha)),
            DwSepConv(int(128*self.alpha), int(256*self.alpha), stride=2),
            DwSepConv(int(256*self.alpha), int(256*self.alpha)),
            DwSepConv(int(256*self.alpha), int(512*self.alpha), stride=2),
            # 5층에서->3개층으로 줄임 CIFAR-10
            *[DwSepConv(int(512 * self.alpha), int(512 * self.alpha)) for _ in range(3)],
            DwSepConv(int(512*self.alpha), int(1024*self.alpha), stride=2),
            DwSepConv(int(1024*self.alpha), int(1024*self.alpha))
        )
        self.classfier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(int(1024*self.alpha), num_classes)
        )
        if init_weight: #초기화 구동함수 호출
            self._initialize_weight()
    def forward(self, x):
        x = self.stem(x)
        x = self.model(x)
        x = self.classfier(x)

        return x
    
    #모델의 초기 Random을 커스터마이징 하기 위한 함수
    def _initialize_weight(self):
        for m in self.modules(): #설계한 모델의 모든 레이어를 순회
            if isinstance(m, nn.Conv2d): #conv의 파라미터(weight, bias)의 초가깂설정
                # Kaiming 초기화를 사용한 이유:
                # Kaiming 초기화는 ReLU 활성화 함수와 함께 사용될 때 좋은 성능을 보임
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d): #BN의 파라미터(weight, bias)의 초가깂설정
                # BatchNorm 레이어의 가중치와 바이어스를 간단한 값으로 초기화
                nn.init.constant_(m.weight, 1) # 1로 다 채움
                nn.init.constant_(m.bias, 0) # 0으로 다 채움

            elif isinstance(m, nn.Linear): #FCL의 파라미터(weight, bias)의 초기값 설정
                # 선형 레이어의 가중치를 정규 분포로 초기화
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```

```python
model = MobileNetV1(alpha = 1)
summary(model, input_size = (2, 3, 224, 224), device = "cpu")
```

## 6. Experiment

Data : CIFAR-10 (논문에서는 ImageNet 데이터를 사용)

Augmentation : RandomCrop, RandomHorizontalFlip, Normalization (논문에서도 많은 증강을 하지 않는다고 함)

Optimizer : RMSprop, momentum=0.9, weight decay=0.0005

Criterion : CrossEntropy Loss

Scheduler : 매 10 에포크마다 학습률 0.9배 감소

### 6.1 설정

|  | ***α*** | ***ρ*** |
| --- | --- | --- |
| 실험 1 | 1 | 224 |
| 실험 2 | 1 | 128 |
| 실험 3 | 0.75 | 224 |
| 실험 4 | 0.75 | 128 |
| 실험 5 | 0.25 | 128 |
| 실험 6 | 0.25 | 64 |

### 6.2 결과

| 설정 | 최대 Training Accuracy | 최대 Test Accuracy | Training Loss (epoch 100) | Test Loss (epoch 100) | Test Error Rate (epoch 100) |
| --- | --- | --- | --- | --- | --- |
| Alpha=1.00, Rho=224 | 78.65% | 74.36% | 0.6356 | 0.8063 | 25.64% |
| Alpha=1.00, Rho=128 | 79.27% | 75.55% | 0.6141 | 0.7506 | 24.45% |
| Alpha=0.75, Rho=224 | **79.34%** | **80.40%** | 0.6132 | 0.5810 | **19.60%** |
| Alpha=0.75, Rho=128 | 79.03% | 77.18% | 0.6128 | 0.6738 | 22.82% |
| Alpha=0.25, Rho=128 | 75.06% | 72.24% | 0.7398 | 0.8192 | 27.76% |
| Alpha=0.25, Rho=64 | 75.36% | 70.42% | 0.7267 | 0.8852 | 29.58% |

<img src="https://github.com/user-attachments/assets/1a0f6fa9-93ee-47cf-a5df-061c948519cd" width="500" height="400">
<img src="https://github.com/user-attachments/assets/a55531eb-8db3-400f-afd8-b3e72bd9f1d7" width="500" height="400">
<img src="https://github.com/user-attachments/assets/549c50af-c007-4a66-ab02-519394f5e158" width="500" height="400">
<img src="https://github.com/user-attachments/assets/9e77da56-3232-4803-b7a8-3bb1bcb66627" width="500" height="400">
<img src="https://github.com/user-attachments/assets/1b5eee2f-2660-480f-bc95-c44795e77ba7" width="500" height="400">
<img src="https://github.com/user-attachments/assets/bb2bc070-935b-4c1f-89ba-0ac42f75bab9" width="500" height="400">


### 6.3 평가

논문에서 기존 Input data인 ImageNet(224 X 224) 데이터를 사용하지 않고 CIFAR-10 (32 X 32)로 진행했기에 논문에서 제시한 네트워크 구조를 변경하였다 (3개층으로 줄인 것, 초기 stride=2를 stride=1로 바꾼 부분)

`alpha=0.75`와 `rho=224` 조합이 전체적으로 가장 높은 테스트 성능을 보였으나, CIFAR-10의 작은 데이터셋에 대하여 지나치게 복잡한 모델을 돌리다 보니 과적합 문제가 보인다.
