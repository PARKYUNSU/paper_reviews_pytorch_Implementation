# DenseNet
---
Link to Paper:

**“Densely Connected Convolutional Networks”** - 2017

— Gao Huang, Zhuang Liu, Laurens Van der Maaten

https://arxiv.org/pdf/1608.06993

---

Table of Contents

1. Introduction
2. DenseNet Architecture
3. Network Architectures
4. Code
5. Agumentation and Training Setup
6. Conclusion

---

## 1. Introduction

ResNet

ResNet은 VGG와 같은 전통적인 Convolution Network의 Architecture를 설계할 때 너무 많은 필터를 거치면 성능이 하락하는 문제를 해결한 모델입니다.

이전 레이어의 출력을 다음 레이어의 출력에 `summnation`하는 방법인 `Skip Connection`을 사용하여 성능 하락 문제를 해결하였습니다. 이를 식으로 나타내면 다음과 같습니다.

$x_{\ell} = H_{\ell}(x_{\ell-1}) + x_{\ell-1}$

하지만 이전 레이어의 출력을 다음 레이어의 출력과 합해서 더한다는 점에서, 정보 흐름이 이전 레이어에서 다음 레이어들로 흘러가는 것을 방해할 수 있다는 점이 있습니다.

DenseNet

DenseNet은 모든 이전 레이어의 출력 정보를 다음 레이어의 입력으로 받아오는 `Dense Connectivity` 방법을 사용하여 논문에서 해결방법을 제안하였습니다.

## 2. DenseNet Architecture

### 2.1 Dense Connectivity

![image](https://github.com/user-attachments/assets/e287e3a1-8ff6-4162-87b4-20c7515bd76b)
Fig.1

`Dense Connectivity`은 이전 레이어의 출력을 다음 레이어의 입력과 연결하는 아이디어는 ResNet과 같지만 `Concatenation`하는 것이 차이점이다.

예를들어, 층 $H_{\ell}$ 에서 나오는 출력이 $x_{\ell}$ 라면 , `Concatenation` 은 $x_0​,x_1​,x_2​,...,x_{\ell-1}$ 를 모두 하나의 벡터로 연결하여 다음 층의 입력으로 사용합니다. 결과인 $x_ℓ$을 수식으로 나타내면 다음과 같습니다.

$x_ℓ = H_ℓ([x_0, x_1, . . . , x_{\ell-1}])$

Fig.1 을 살펴보자. (growh rate = 4)

1. Input에 6개의 채널을 가진 Feature Map이 있다고 가정 (빨간색)
2. Bn-ReLU-Conv를 거치면서 4개의 Feature Map을 뽑음(초록색)
3. 6 + 4 개의 Feature Map이 Bn-ReLU-Conv를 통과해서 4개의 Feature Map을 뽑음
    
    6 (빨간색) + 4 (초록색) → Bn-ReLU-Conv → 4 (보라색)
    
4. 6 + 4 + 4 개의 Feature Map이 Bn-ReLU-Conv를 통과해서 4개의 Feature Map을 뽑음
    
    6 (빨간색) + 4 (초록색) + 4 (보라색) → Bn-ReLU-Conv → 4 (노란색)
    
5. 6 + 4 + 4 + 4 개의 Feature Map이 Bn-ReLU-Conv를 통과해서 4개의 Feature Map을 뽑음
    
    6 (빨간색) + 4 (초록색) + 4 (보라색) + 4 (노란색) → Bn-ReLU-Conv → 4 (주황색)
    

Fig.1 처럼 모든 층이 이전의 모든 층과 연결되어 훨씬 더 촘촘한 연결 구조를 가지며, 장점으로 정보의 손실이 적으며, 각 층이 이전 층의 모든 Feature Map을 직접적으로 접근할 수 있어 학습 과정에서 더 풍부한 정보 제공가능해졌습니다.

`Dense Connectivity`를 활용하여 각 레이어를 서로 연결하고, 이렇게 연결된 레이어 묶음을 하나의 Block으로 구성하였고, 이러한 블록들을 이어붙여서 하나의 모델로 만들었습니다.

### 2.2 Composite function

Identity mappings in deep residual networks. - 2016 논문에서 착안하여, Composite function 인 

$H_ℓ$(·)을 정의 ⇒ Batch Normalization (BN) → ReLU → 3 X 3 Convolution (Conv)

Fig.2 에서 Origin Block 기존 순서인 Conv → BN → ReLU를 ⇒  Composite Function인 BN → ReLU → Conv 변형해서 사용

<img src="https://github.com/user-attachments/assets/1c928f59-cc82-43ad-876d-4be8e5641fa3" width="600" height="400"/>

Fig.2

### 2.3 Pooling Layers
![image](https://github.com/user-attachments/assets/bcb9e020-3a90-4a60-b9d8-0bb125884683)
Fig.3

Concatenation 방법으로 이어 붙인 모든 Block들은 같은 size 여야 합니다. 그러나 여러 layer들을 지나면서 Feature Map의 수가 점점 증가합니다. 이때 Transition Layer를 사용해서 Feature Map의 크기를 줄여 모델의 메모리 사용 및 계산 복잡성을 조절하는데 필수적입니다.

논문에서 Transition Layer은

Batch Normalization → 1 X 1  Convolution Layer → 2 X 2 Average Pooling Layer

로 구성 되어있습니다.

### 2.4 Growth rate

Dense Block 에서 이전 레어이의 Block들의 출력을 입력값으로 Concatenation 하여 받아오기로 하였습니다.

그러면, 만약 $H_ℓ$ 이 $k$개의 출력을 만들어낸다면 $ℓ^{th}$ layer은 

$k_0 + k × (l-1)$ 개의 입력 Feature Maps를 받아오게 됩니다.

→ $k_0$(First feature channel) + $k$(Growth rate) X $(l-1)$(num layers) = Last Feature channel

Densent은 Concatenation 방법으로 Feature Channel이 점점 더 커지기 때문에 그 크기를 제한하는 Growth rate를 제안합니다.

DenseNet에서 Growth rate는 각 레이어가 얼마나 많은 새로운 정보를 추가할지를 결정해줍니다. 즉, $k$라는 Growth rate는 각 레이어가 생성사는 새로운 Feature map의 수를 의미하며, 이 값이 크면 레이어가 추가하는 정보가 많아지고 작으면 적어집니다.

### 2.5 Bottleneck layers

<img src="https://github.com/user-attachments/assets/72a1b61a-c3f0-43c0-83e3-1775c9fac760" width="600" height="400"/>

Fig.4

ResNet과 Inception 등에서 사용된 Bottleneck layer은 DenseNet 에서도 사용되었습니다.

3x3 convolution 전에 1x1 convolution을 거쳐서 입력 Feature map의 channel 개수를 줄이는 것 까지는 같으나, 다시 입력 Feature map channel 수 만큼 생성하는 Growth rate 만큼의 Feature map을 생성하는 것이 차이점이며, 이를 통해 Computational cost를 줄일 수 있습니다.

$H_ℓ$ = BN → ReLU → 1 X 1 Conv, 4*k → BN → ReLU → 3 X 3 Conv, k   의 형식

또한 구현할 때 약간의 특이한 점이 존재합니다. DenseNet의 Bottleneck Layer는 1x1 Convolution 연산을 통해 4*Growth rate 개의 Feature map을 만들고 그 뒤에 3x3 Convolution을 통해 Growth rate 개의 Feature map으로 줄여주는 점이 특이합니다. Bottleneck layer를 사용하면, 사용하지 않을 때 보다 비슷한 Parameter 개수로 더 좋은 성능을 보임을 논문에서 제시하고 있습니다.

### 2.6 Compression

모델의 압축성을 개성하기 위해서, Transition Layer 에서 Feature map의 수를 줄일 수 있습니다.

Transition Layer에서, Dense Block이 m개의 Feature map를 가지고 있다면, 압축 정도를 결정하는 0 ~ 1 사이인 Theta를 하나 정해서 $θm$ 개의 Feature map을 만듭니다.

Theta = 1 일 경우 Transition Layer 를 지나는 경우 Feature map은 변하지 않게 되며, DenseNet 에서는 Theta를 1보다 작게하는 것을 추천하며, 논문에서는 0.5로 사용 하였습니다.

## 3. Network Architectures

![image](https://github.com/user-attachments/assets/88b30d65-18a7-4e54-aab5-531d96b652dc)
Table.1

Table.1을 보면 k는 32와 48로 지정하고, Dense Block은 Batch Norm → ReLU → Conv 구조로 되어있다.

DenseNet 121를 예시로

Input conv, pooling : 1

Dense bolck(1) : 2X16=12

Transition : 1

Dense bolck(2) : 2X12=24

Transition : 1

Dense bolck(3) : 2X24=48

Transition : 1

Dense bolck(3) : 2X16=32

Classification : 1

1 + 12 + 1 + 24 + 1 + 48 + 1 + 32 + 1 = 121 이 된다.

## 4. Code

**BottleNeck**

```python
class bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(bottleneck, self).__init__()
        self.k = growth_rate
        self.res = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4*self.k, kernel_size=1, bias=False),
            nn.BatchNorm2d(4*self.k),
            nn.ReLU(),
            nn.Conv2d(4*self.k, self.k, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([self.res(x), x], dim=1)
```

<img src="https://github.com/user-attachments/assets/5a35b242-682e-4043-8552-cf7548ad188d" width="550" height="420"/>


**Transition**

```python
class transition(nn.Module):
    def __init__(self, in_channels):
        super(transition, self).__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, int(in_channels/2), kernel_size=1, bias=False), 
            nn.AvgPooling2d(2)
        )
    def forward(self, x):
        return self.trans(x)
```

<img src="https://github.com/user-attachments/assets/b5af0c43-ff1f-4e61-97ee-b452adfd44af" width="600" height="380"/>


**DenseNet**

```python
class DenseNet(nn.Module):
    def __init__(self, block_list, growth_rate, num_classes = 1000):
        super().__init__()

        assert len(block_list) == 4

        self.k = growth_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 2 * self.k, 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(2 * self.k),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, stride = 2, padding = 1)

        self.dense_channels = 2 * self.k
        dense_blocks = []
        dense_blocks.append(self.make_dense_block(block_list[0]))
        dense_blocks.append(self.make_dense_block(block_list[1]))
        dense_blocks.append(self.make_dense_block(block_list[2]))
        dense_blocks.append(self.make_dense_block(block_list[3], last_stage = True))
        self.dense_blocks = nn.Sequential(*dense_blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.dense_channels, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.dense_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def make_dense_block(self, num_blocks, last_stage = False):
        layers = []
        for _ in range(num_blocks):
            layer = bottleneck(self.dense_channels, self.k)
            layers.append(layer)
            self.dense_channels += self.k

        if last_stage:
            layers.append(nn.BatchNorm2d(self.dense_channels))
            layers.append(nn.ReLU())
        else:
            layers.append(transition(self.dense_channels))
            assert self.dense_channels % 2 == 0
            self.dense_channels //= 2
            
        return nn.Sequential(*layers)
```

```python
def DenseNet121():
    return DenseNet(block_list=[6, 12, 24, 16], growth_rate=32)

def DenseNet169():
    return DenseNet(block_list=[6, 12, 32, 32], growth_rate=32)

def DenseNet201():
    return DenseNet(block_list=[6, 12, 48, 32], growth_rate=32)

def DenseNet264():
    return DenseNet(block_list=[6, 12, 64, 48], growth_rate=32)
```

**DenseNet121의 출력 변화 과정**

![image](https://github.com/user-attachments/assets/86c205dd-d88e-4c04-ad2e-8766189b1904)

**DenseNet-121 Parameters**

<img src="https://github.com/user-attachments/assets/fbee205f-afa7-4df0-b1ee-c653ca9a6fac" width="600" height="200"/>

**ResNet-101 Parameters**

<img src="https://github.com/user-attachments/assets/0863295b-b5bd-4d84-a7d7-d6ee0bda9310" width="600" height="200"/>

DenseNet 이 레이어가 더 많아도 ResNet 보다 연산량이 적음

## 5. Agumentation and Training Setup

### 1) Agumentation

Random Cropping - 원본 이미지를 무작위로 잘라내어 학습

Horizontal Flip - 이미지 수평으로 뒤집기

Color Jittering - 이미지의 밝기, 대비, 채도 등을 무작위로 변화

Mean Subtraction - 원본 이미지에서 각 픽셀 값에서 해당 픽셀의 평균 값을 빼서 정규화

### 2) Hyper parameters

Optimizer - SGD

Batch_size - 64

weight decay - 0.001, momentum - 0.9

Learning_rate - 0.1 로 시작해서 전체 에포크의 50% 75% 지점에서 학습률을 0.1배로 줄임

epoch - 300

### 3) Dataset

CIFAR-10

## 6. Conclusion

[Training] loss: 1.5326, accuracy: 0.4452: 100%|██████████| 782/782 [00:56<00:00, 13.84it/s]
100%|██████████| 157/157 [00:06<00:00, 23.54it/s]

epoch 001, Training loss: 1.5326, Training accuracy: 0.4452
Test loss: 1.4567, Test accuracy: 0.4688

[Training] loss: 0.9895, accuracy: 0.6540: 100%|██████████| 782/782 [00:57<00:00, 13.68it/s]
100%|██████████| 157/157 [00:06<00:00, 23.37it/s]

epoch 010, Training loss: 0.9895, Training accuracy: 0.6540
Test loss: 0.9893, Test accuracy: 0.6517

[Training] loss: 0.7787, accuracy: 0.7293: 100%|██████████| 782/782 [00:58<00:00, 13.44it/s]
100%|██████████| 157/157 [00:06<00:00, 23.62it/s]

epoch 020, Training loss: 0.7787, Training accuracy: 0.7293
Test loss: 0.8923, Test accuracy: 0.6978

[Training] loss: 0.6836, accuracy: 0.7625: 100%|██████████| 782/782 [00:56<00:00, 13.81it/s]
100%|██████████| 157/157 [00:06<00:00, 23.88it/s]

epoch 030, Training loss: 0.6836, Training accuracy: 0.7625
Test loss: 0.7832, Test accuracy: 0.7325

…

…

…

[Training] loss: 0.4749, accuracy: 0.8357: 100%|██████████| 782/782 [01:00<00:00, 12.97it/s]
100%|██████████| 157/157 [00:07<00:00, 21.38it/s]

epoch 250, Training loss: 0.4749, Training accuracy: 0.8357
Test loss: 0.6556, Test accuracy: 0.7878

[Training] loss: 0.4736, accuracy: 0.8372: 100%|██████████| 782/782 [01:01<00:00, 12.80it/s]
100%|██████████| 157/157 [00:07<00:00, 22.12it/s]

epoch 260, Training loss: 0.4736, Training accuracy: 0.8372
Test loss: 0.5801, Test accuracy: 0.8075

[Training] loss: 0.4811, accuracy: 0.8335: 100%|██████████| 782/782 [01:00<00:00, 12.91it/s]
100%|██████████| 157/157 [00:07<00:00, 21.72it/s]

epoch 270, Training loss: 0.4811, Training accuracy: 0.8335
Test loss: 0.5839, Test accuracy: 0.8006

[Training] loss: 0.4677, accuracy: 0.8391: 100%|██████████| 782/782 [00:58<00:00, 13.31it/s]
100%|██████████| 157/157 [00:06<00:00, 22.62it/s]

epoch 280, Training loss: 0.4677, Training accuracy: 0.8391
Test loss: 0.6147, Test accuracy: 0.7985

[Training] loss: 0.4746, accuracy: 0.8360: 100%|██████████| 782/782 [01:00<00:00, 12.91it/s]
100%|██████████| 157/157 [00:06<00:00, 22.68it/s]

epoch 290, Training loss: 0.4746, Training accuracy: 0.8360
Test loss: 0.6063, Test accuracy: 0.7927

[Training] loss: 0.4692, accuracy: 0.8378: 100%|██████████| 782/782 [01:00<00:00, 12.94it/s]
100%|██████████| 157/157 [00:06<00:00, 22.48it/s]

epoch 300, Training loss: 0.4692, Training accuracy: 0.8378
Test loss: 0.6148, Test accuracy: 0.7959
Model saved to [/kaggle/working/data/model.pth](https://file+.vscode-resource.vscode-cdn.net/kaggle/working/data/model.pth)

![image](https://github.com/user-attachments/assets/3da1f781-fb45-42d7-948c-c1df38f2b8ce)
