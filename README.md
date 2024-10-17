# SE-Net
 ILSVRC 2017 ImageNet 분류 대회에서 1위 
SNet(Squeeze and Excitation Networks)
저자 : Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

## Squeeze and Excitation Networks

### 1.Introduction
**Squeeze and Excitation Networks(SENet)**은 기존 CNN의 성능을 향상시키기 위해 간단하고도 효과적인 방법을 제안한 네트워크입니다.

CNN 과정에서 중요한 정보와 덜 중요한 정보를 구별하는 능력을 높여주는 SE Block을 도입함으로써 채널 간의 상호작용을 학습하고 각 채널의 중요도를 조정합니다.

논문에서는 SE Block을 다양한 CNN 모델 구조에 통합하여, SE Block의 네트워크 표현력의 향상 효과에 대해서 설명하고 있습니다. 그 결과로 ImageNet 대회에서 큰 성과를 거두며 그 유용성을 입증했습니다.


<img src="https://velog.velcdn.com/images/qkrdbstn24/post/bfded920-3176-43f9-8e43-14d1e17cc153/image.png" alt="image" width="400"/> Fig.1

먼저, CNN(인공신경망)은 **Local Receptive Field** 을 기반으로 연산을 한다는 점에서 시작됩니다.

CNN의 연산 과정을 생각하면, 위의 Fig.1 처럼 이미지의 작은 부분, 즉 국소영역에만 집중해서 추출합니다. 이때 각 Convolution filter 또는 Feature map은 이미지의 일정한 작은 영역에서(local)을 학습하여 정보를 얻습니다. 이러한 CNN의 연산과정은 이미지를 작은 크기의 패치로 분해하고 그 패치에서 중요한 특징(Edge, 코너, 질감 등)을 뽑아냅니다. 

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/aaf5f67b-dd33-47b7-a4c5-76a52c001228/image.png" alt="image" width="400"/> Fig.2


그러나, 논문에서는 CNN의 연산은 **국소적 영역(local)**에서 정보를 얻기 때문에 **전체적인(global)문맥**을 이해하기는 어렵다라는 문제점을 제기합니다. 즉, Feature map도 각 필터의 국소적 시야에서 추출된 정보로, 이미지의 특정 영역에만 집중합니다.
 예를들어, 한 이미지에서 동물의 귀에 집중하게 되면, 그 귀가 고양이의 귀인지 강아지의 귀인지 구분하기 어려울 수 있습니다. 그러나 이미지 전체를 보면 고양이인지 강아지인지 쉽게 판단할 수 있습니다.
 
논문의 저자들은 다음과 같이 얘기합니다.

 **"We propose a mechanism that allows the network to perform feature recalibration, through which it can learn to use global information to selectively emphasise informative features and suppress less useful ones"**

SE Block을 도입하면서, 네트워크가 전체적인 정보를 사용하여 중요한 특징을 강조하고, 덜 중요한 특징을 사용 억제하여 **특징을 재보정**하는 메커니즘을 제안합니다.


### Squeeze and Excitation Blocks


<img src="https://velog.velcdn.com/images/qkrdbstn24/post/6ae36879-9a07-4f87-88d8-4a19cb1f5f05/image.png" alt="image" width="700"/>


**$F_{tr}$** 은 단순한 Convolution 연산입니다.

$F_{tr}$ : $X$ $→$ $U$, $X$ $∈$ $R^{H^′×W^′×C^′}$, $U$ $∈$ $R^{H×W×C}$

$X$ = $[x^1,x^2,...,x^{C^′}]:$ Input

$U$ = $[u_1,u_2,...,u_C]$ : $F_{tr}$을 통과한 Output

$V$ = $[v_1,v_2,...,v_c]$ : 필터 집합 $V$와 각 필터의 파라미터. $v_c$는 $c$번째 필터의 파라미터

$v_c$ = $[v^1_c,v^2_c,...,v^{C^′}_c]$ : $v_c$ 가중치 벡터, $v^s_c$는 $s$번째 입력 채널에서 $c$번째 출력 채널로 연결되는 2D spatial kernel이자 학습가능한 가중치


<img src="https://velog.velcdn.com/images/qkrdbstn24/post/a10bf854-ccf7-4e2c-82d7-02ec9cacf411/image.png" alt="image" width="500"/>


수식은, $C'$번째 출력 채널 값이 여러 입력 채널의 가중합으로 계산됩니다. 각 입력 채널 값에 가중치를 곱해주고, 그 결과를 모두 더해서 하나의 출력 채널을 생성하는 Convolution 연산 과정입니다.

### Squeeze: Global Information Embedding
Squeeze Operatioin은 말그대로 압축하는 과정입니다. Feature Map을 각 Channel로 분리하여 중요한 정보만 추출해서 사용하는 것 입니다.


논문에서는 GAP(Global Average Pooling)로 중요 정보를 추출 합니다. GAP를 사용하여 Global Spatial Information을 Channel Descriptor로 압축시킬 수 있습니다.

수식은 다음과 같습니다.

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/a8359671-9e5b-41f1-9157-b871dcef84c3/image.png" alt="image" width="500"/>

$z_c$ : **채널 $c$**에 대한 Squeeze 단계 출력 값

$H$ , $W$ : 입력 Feature Map의 높이와 너비

$u_c(i, j)$ : $c$채널의 **위치$(i, j)$** 값

$\frac{1}{H \times W}$ : 평균을 위한 나누기


입력 Feature Map의 각 채널에 대해 spatial dimensions에서 평균을 계산합니다. 주어진 채널의 모든 픽셀 값들을 더한 후, 픽셀 수 **$H \times W$**로 나누어 평균을 구합니다. 이를 통해서 공간적인 정보를 요약하여 각 채널의 Global Representation을 얻습니다. GAP를 통해 전체적인 공간적 패턴을 요약할 수 있어 네트워크가 특정 Channel 정보가 얼마나 중요한지를 학습할 수 있게 도와줍니다.

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/719ae51c-b68e-4944-8b32-5dd4c48167c5/image.png" alt="image" width="400"/>

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/e9596459-e1a9-46e6-b1cf-0fcaea189518/image.png" alt="image" width="400"/>

논문의 저자들은 간단한 방식으로 GAP를 사용했지만, 다른 방법론을 사용할 수 있다고 합니다.

### Excitation: Adaptive Recalibratcion
Excitation Operation은 재조정 과정으로, 채널 간 의존성(Channel-wise-dependencies)을 계산하게 됩니다. 논문에서는 Fully Connected Layer와 비선형 함수를 조절하는 것으로 간단하게 계산합니다. 수식은 다음과 같습니다.

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/e9a21613-425a-46bd-a9b6-f94d76779884/image.png" alt="image" width="500"/>

$s_c$ : **채널 $c$**에 대한 Excitation 가중치(Excitation 단계의 출력 값)

$z$ : Squeeze 단계에서 구한 Channel Descriptor

$W_1$ , $W_2$ : Fully Connected Layer의 가중치 행렬. $W_1$은 채널 수를 줄이며 $W_2$는 채널 수를 다시 원래대로 복원

$δ$ : ReLU 활성화 함수

$σ$ : Sigmoid 활성화 함수

그림으로 간단하게 그려보면 다음과 같습니다.(그림은 C=4, r=2라 가정한 그림)

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/bef315af-2d1a-4945-9421-88e592b4316c/image.png" alt="image" width="600"/>

1) Squeeze 단계에서 생성된 채널 디스크립터 $z$를 입력으로 받아, $W_1$의 노드 **C**를 reduction ratio **r**을 통해서 노드 수를 줄입니다. 이때 차원 축소를 통해 비선형성을 도입하고, 중요한 특징만을 더 잘 학습할 수 있도록 합니다.
   
2) $W_{1}z$ 에 ReLU를 적용하여 음수 값을 0으로 비선형 변환을 수행합니다.
   
3) $W2$에서 다시 피쳐맵의 수 **C**만큼 복원합니다. 이 단계는 각 채널에 대한 가중치를 결정하는 과정입니다.
   
4) 최종적으로 Sigmoid 함수를 사용하여 0~1 사이의 값으로 정규화하여 각 채널의 가중치를 확률적으로 해석할 수 있으며, 중요한 채널은 가종하고 덜 중요한 채널은 억제할 수 있습니다.

이렇게 모든 함수를 거쳐서 나온 값을 아래 수식으로 계산합니다.

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/7c4b2b37-5a33-46c7-8648-3c16ae552570/image.png" alt="image" width="500"/>

$u_c$ : 입력 특징 맵의 $c$번째 채널. Squeeze 단계 전에 Convolution 연산으로 나온 $c$번째 채널의 Spatial 특징

$s_c$ : Excitation 과정에서 학습된 $c$번째 채널에 대한 중요도. Sigmoid로 정규화 되어 0 ~ 1 값으로 채널의 중요도를 나타낸다.

$\widetilde{x}_c$ : Reweighting 과정의 출력, 입력 특징맵 $u_c$에 가중치 $s_c$를 곱한 값

$F_scale$ : Channel-wise Multiplication(채널 별 곱셈), 각 채널의 Global Spatial에 걸쳐 동일한 가중치 $s_c$를 곱하여 채널의 중요도 반영


1) 입력 Feature map $u_c$는 $c$번째 채널에서 추출된 Spatial Information을 포함하고 있습니다.
   
2) Excitaion 단계에서 학습된 가중치 $s_c$는 해당 채널의 중요도를 나타내며, 중요도가 높은 채널은 강조되고, 중요도가 낮으면 억제됩니다.
 
3) Reweighting 단계에서, 각 채널에 대해 $s_c$를 곱하여, 중요도에 따라 입력 Feature map을 재조정합니다. $s_c$가 1에 가까우면 정보가 유지되며, 반대로 0에 가까우면 해당 채널의 정보는 많이 억제됩니다.

### Exemplars: SE-Inception and SE-ResNet
논문 저자들은 SE Block을 VGGNet, InceptionNet, ResNet등 다양한 모델에 적용을 합니다. InceptionNet과 ResNet에 적용한 구조는 다음과 같습니다.

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/1b23721a-4975-4ecb-8688-45158cbc264d/image.png" alt="image" width="400"/>

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/abfb7658-0cd8-4e96-b906-b9775c90f1ba/image.png" alt="image" width="400"/>

ResNet-50과 ResNeXt-50을 예시로 SEBlock을 적용했을때 모델 구조의 설명

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/a3d2e35e-f485-423d-a5ab-d8d13f7036e6/image.png" alt="image" width="700"/>


SEBlock을 다양한 모델에 적용했을때 모델의 성능 비교 테이블

<img src="https://velog.velcdn.com/images/qkrdbstn24/post/ec9f400b-ba9a-4c80-b342-a21fd6ee7ea3/image.png" alt="image" width="700"/>

SEBlock을 적용한 것 만으로도 연산량은 아주 조금 늘어났지만 모델의 성능을 어느정도 상승시키는 특징을 가지고 있다.


### Pytorch 코드 구현
#### SEBlock

SEBlock은 앞서 얘기한 계산 과정처럼 GAP와 ReLU, Sigmoid의 연산과정으로 나타낸다. 다음은 SEBlock의 Pytorch 코드


```python
import torch
from torch import nn

class SE_block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SE_block, self).__init__()

        # Squeeze: Global Information Embedding
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        
        # Excitation: Adaptive Recalibration
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        se_weight = self.squeeze(x)  # (N, C, H, W) -> (N, C, 1, 1)
        se_weight = se_weight.view(se_weight.size(0), -1)  # (N, C, 1, 1) -> (N, C)로 reshape
        se_weight = self.excitation(se_weight)  # 채널별 중요도 계산
        se_weight = se_weight.view(se_weight.size(0), se_weight.size(1), 1, 1)  # (N, C) -> (N, C, 1, 1)로 다시 reshape
        return x * se_weight  # 입력에 중요도를 반영하여 조정
```


![](https://velog.velcdn.com/images/qkrdbstn24/post/097baef0-27ca-495c-b365-e8bf2ed7da1b/image.png)

