
Link to Paper:

**“Efficient Convolutional Neural Networks for Mobile Vision Applications”** 

— Andrew G. Howard Weijun Wang, Menglong Zhu Tobias Weyand, Bo Chen Marco Andreetto, Dmitry Kalenichenko Hartwig Adam

https://arxiv.org/pdf/1704.04861

---

Table of Contents

1. Introduction
2. 

---

모바일 및 임베디드 비전 어플리케이션을 위한 효율적인 모델 MobileNet

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ed949dfc-9e1b-48c7-aebd-97b360602a23/78ad2124-be69-4e6c-aca5-c9d2b3ed13a4/image.png)

이 모델은 Object Detection, Face Attributes, Finegrain Classification, Landmark Recognition 사용에 효과적임을 보여줌

MobileNet 논문에서 제공한 기법은 3가지 기법

**1) Depthwise Separable Convolutions (깊이 분리 합성곱)**

**2) Width Multiplier(*α*)**

**3) Resolution Multiplier(*ρ*)**

Depthwise Separable Convolutions

Depthwise Separable Convolutions 기법은 합성곱을 깊이 별 합성곱과 1X1 합성곱으로 나누는 형태의 Factorized Convolution

즉, 합성곱의 연산과정을 2단계로 분해하는 방법

이 과정으로 연산 비용을 감소효과를 가져옴 

![fig.1](https://prod-files-secure.s3.us-west-2.amazonaws.com/ed949dfc-9e1b-48c7-aebd-97b360602a23/b859f757-bb7a-41cd-b572-b9fd103a091e/image.png)

fig.1

fig.1을 기준으로 `in_feature` 의 shape는[$M , D_F, D_F$]

`Conv Layer`의 커널 사이즈 [$D_K, D_K$], $N$ 개 로 정의 했을 때, 그 수식은

$D_K · D_K · M · N · D_F · D_F$

논문에서는, 기존 연산 과정을 Depth wise(공간 축), Point wise(채널 축) 으로 나눠서 계산

![fig.2](https://prod-files-secure.s3.us-west-2.amazonaws.com/ed949dfc-9e1b-48c7-aebd-97b360602a23/75379404-7867-4531-9581-f9b7864c3980/image.png)

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
=\frac1 N + \frac 1 {D^2_k}
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

결과적으로 계산량은 :

$=\frac{29,376} {1,769,472}
≈0.0167$

$\frac1 N + \frac 1 {D^2_k}
=\frac1 {64} + \frac1 {32^2} ≈0.0167$
