# Wide Resnet

---

Link to Paper:

**“Wide Residual Networks”** - 2016

— Sergey Zagoruyko, Nikos Komodakis

https://arxiv.org/pdf/1605.07146

---

Table of Contents

1. Introduction
2. Width and Depth
3. Weakness of ResNet
4. Wide Residual Networks
5. Strength of Wide ResNet
6. Code
7. Experimental Results

---

## 1. Introduction

너비(Width)와 깊이(Depth)의 문제는 오랫동안 머신러닝에서 논의 되어왔습니다. 네트워크의 깊이를 깊게하여 학습하면서 더 나은 성능이 나옴에 따라 어떻게 더 깊은 망을 설계하는지 연구 해왔습니다.  그러나 망이 깊어지면서 성능 저하 문제 및 기울기 소실 문제로 깊은 학습이 어려워졌기에 새로운 연구가 필요했습니다.

그 연구로 VGGNet, Inception, ResNet 등의 모델이 발표되었으며 네트워크의 Depth에 관한 업적을 쌓았습니다. 

그러나 Wide Residual Networks는 깊이(Depth)에 중점을 두지 않고 너비(Width)에 집중하여 ResNet 모델의 한계를 극복하려 했습니다.

이 논문에서는 더 넓은 Residual Block을 사용하여서 성능 향상에 집중하였습니다.

## 2. Width and Depth

<img src="https://github.com/user-attachments/assets/4dce1dd4-1c1e-409f-b47f-570e00c0f90a" width="700" height="400">

Fig.1


![image](https://github.com/user-attachments/assets/8a666050-74f0-4694-8773-0c39646c27e4)
Fig.2

Fig.1을 기준으로 Width 란 Convolution Filter 개수 또는 Output Feature Map의 채널 수를 의미하며, Width가 넓은 Network란 Convolution Filter 개수를 많이 사용하여 Output Feature Map의 채널 수가 많은 모델을 의미합니다.

반면, Fig.2를 기준으로 CNN을 구성하는 Convolution, Activation Function, Pooling 등의 연산이 하나의 층을 이루며, 이 층의 개수를 Depth 로 표현됩니다. Depth 가 깊은 모델이란 이런 연산들이 많이 구성된 Network를 의미합니다.

깊이에 관한 연구들을 대표하는 모델로 `VGGNet`, `Inception`, `ResNet`이 있습니다.

<img src="https://github.com/user-attachments/assets/075cb46e-1f7d-4100-964e-2cfafbc3e960" width="800" height="550">



`VGGNet`은 3X3 Conv를 이용하여 깊은 Network를 구성

`Inception`은 1X1 Conv, 3X3 Conv, 5X5 Conv, Pooling을 조합하여 깊은 Network를 구성

`ResNet`은 Skip Connection을 사용하여 Network를 더 깊게 쌓았습니다.

## 3. Weakness of ResNet

### 1) Circuit complexity theory

ResNet에서는 수천 개의 레이어를 쌓아도 성능이 하락하지 않고 향상되었습니다. 그러나 성능이 몇 퍼센트 향상될 때마다 필요한 레이어의 수가 거의 2배로 늘어나기 때문에, 깊은 Network를 훈련하는 데에 효율이 떨어지는 문제가 발생됩니다.

### 2) Diminishing feature reuse

ResNet에서 깊은 Network 훈련을 가능하게 도와준 Residual Block이 논문에서 문제점으로 거론되었습니다. Network를 통해 gradient가 흐를 때, Residual Block의 가중치를 통해 흐르도록하는 요소가 없어 아무것도 학습하지 않을 수 있습니다. 그로 인해 유용한 표현을 학습하는 블록이 몇 개 밖에 없거나, 또는 여러 블럭이 최종 목표에 매우 적은 기여만 하여 학습 될 수 있습니다. 이 문제를 Diminishing feature reuse 로 얘기하고 있습니다.

## 4. Wide Residual Networks

논문에서는 ResNet의 문제점을 거론 하면서 깊이에 집중하지 않고 너비에 집중하여 Residual Block을 개선하는 방법을 선택합니다.

$$
x_{l+1} = x_l +F(x_l
,W_l) 
$$

Where:

Ouput : $x_{l+1}$ ($l^{th}$ : $l$  번쨰)

Input : $x_{l}$ ($l^{th}$ : $l$  번쨰)

Residual Function  : $F$

Parameters of the block : $W_l$

ResNet의 `Skip Connections` 또는 `Shortcut Connections` 방법을 사용하였습니다.

<img src="https://github.com/user-attachments/assets/8dd3002b-5903-49b8-868d-1626fa6b27d2" width="400" height="450">



### **1) Layer Width (widening factor k)**:

Wide ResNet은 레이어의 필터 수를 증가시키는 방식으로 네트워크의 너비를 확장합니다. 이때, 너비는 'widening factor'인 k를 사용하여 결정됩니다. 예를 들어, k=2이면 각 레이어에서 사용하는 필터의 수가 기본 ResNet보다 두 배 많습니다. 이러한 확장은 모델이 더 다양한 특징을 학습할 수 있도록 도와줍니다.

### **2) Shortcut Connections**:

Wide ResNet은 ResNet과 마찬가지로 Shortcut Connections를 활용합니다. 이러한 연결은 정보가 네트워크를 더 쉽게 통과할 수 있도록 도와주며, 기울기 소실 문제를 완화시켜 학습을 더욱 안정적으로 만듭니다. 또한, 이러한 연결은 깊이가 깊어질 때 발생하는 문제들을 해결할 수 있게 합니다.

## 5. Strength of Wide ResNet

### **1) 효율적인 학습**:

Wide ResNet은 너비(Width)를 증가시키면서 네트워크의 표현력을 향상시켰습니다. 이를 통해 네트워크가 학습해야 하는 레이어의 수가 줄어들어, 네트워크가 깊어질수록 학습이 어려워지는 기존의 문제를 피할 수 있습니다.

![image](https://github.com/user-attachments/assets/8a592ba9-76b6-47f6-946c-0c52033f5063)


### **2) Feature Reuse**:

깊은 네트워크에서 발생하는 Diminishing Feature Reuse 문제를 완화하기 위해, Wide ResNet은 네트워크의 너비를 늘림으로써 더 많은 피처 맵을 학습합니다. 이러한 방식으로 각 블록이 다양한 특징을 더 잘 학습하고, 이로 인해 최종 성능에 더 크게 기여할 수 있게 됩니다.

### 3) 성능:

여러 연구에서 Wide ResNet은 기존의 ResNet보다 다양한 이미지 분류 작업에서 더 나은 성능을 보였습니다. 특히, Wide ResNet은 복잡한 네트워크를 필요로 하는 데이터셋에서도 안정적으로 학습할 수 있으며, 더 적은 깊이로도 비슷하거나 더 나은 성능을 보여줍니다.

<img src="https://github.com/user-attachments/assets/1e5401d6-f3bd-4046-b9ce-e9dcecf78ab2" width="600" height="400">

## 6. Code

**Basic Block Code**

```python
class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropRate):
        super(Basicblock, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.droprate = dropRate
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity =  x
        x = self.residual(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)
        x += self.skip(identity)
        return x
```

<img src="https://github.com/user-attachments/assets/d908c407-9da9-4b07-89fc-5b41a83312c6" width="800" height="350">


<img src="https://github.com/user-attachments/assets/75f76282-b790-474f-8d97-f454b304b3e1" width="800" height="350">


논문에서는 첫번째 Learning Rate 감수한 후 손실과 검증오류가 갑자기 증가하고, 다음 Learning Rate 감소 까지 높은 값으로 진동하는 발견했습니다. 이 현상이 Weight Decay(가중치 감소)로 인한 현상이고, 이 현상을 Drop out 으로 효과로 제거가 가능하다고 합니다. 

CIFAR 데이터는 0.3 SVHN 데이터는 0.4 Drop out rate를 사용

### Drop out

<img src="https://github.com/user-attachments/assets/91a2f3da-e975-4a5b-b921-e4b08af2773a" width="600" height="300">

Drop-out은 서로 연결된 연결망(layer)에서 0부터 1 사이의 확률로 뉴런을 제거(drop)하는 기법입니다. Drop-out rate를 0.5로 가정할때, Drop-out 이전에 4개의 뉴런끼리 모두 연결되어 있는 Fully Connected Layer에서 4개의 뉴런 각각은 0.5의 확률로 랜덤히게 제거됩니다.

**Drop out 장점**

1. 과적합 방지: 드롭아웃은 일반화(generalization)된 모델을 만들기 위해 사용되기 때문에, 과적합을 방지하는데 효과적입니다.
2. 앙상블 효과: 드롭아웃은 여러 모델을 합친 앙상블 효과를 가집니다. 학습할 때마다 랜덤하게 선택된 뉴런을 사용하므로, 여러 모델을 합친 효과와 유사하게 동작합니다.
3. 계산 효율성: 드롭아웃을 통해 모델이 더 간결해지고, 파라미터 개수가 줄어듭니다. 이는 연산량이 감소하고, 학습 속도가 빨라질 수 있는 장점으로 작용합니다.

**Drop out 단점**

1. 학습 시간: 드롭아웃을 적용하는 것은 추가적인 계산이 필요하기 때문에, 학습 시간이 늘어날 수 있습니다. 
2. 튜닝 파라미터: 드롭아웃 비율 뿐만 아니라, 다른 하이퍼파라미터들도 조정해야할 수 있습니다. 모델에 적합한 드롭아웃 비율을 찾기 위해 여러 실험을 진행하고 튜닝해야할 수 있습니다.

### Depth

<img src="https://github.com/user-attachments/assets/a3177603-1398-4ded-be34-19475e240454" width="600" height="300">

총 Depth 계산

$$
depth=1(conv1)+2×n(conv2)+2×n(conv3)+2×n(conv4)+1(bn + ReLU)
$$

$$
depth=1+2n+2n+2n+1=1+6n+1=6n+2
$$

그러나 여기에 마지막 레이어 그룹에 (BatchNorm, ReLu), Fully Connected 레이어 까지 2개가 더해져서

$$
depth=6n+4
$$

**WideResNet Code**

```python
class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropRate):
        super(WideResNet, self).__init__()
        self.in_channels = 16

        assert ((depth-4) % 6 == 0)
        n = (depth-4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(Basicblock, nStages[1], n, stride=1, dropRate=dropRate)
        self.layer2 = self._wide_layer(Basicblock, nStages[2], n, stride=2, dropRate=dropRate)
        self.layer3 = self._wide_layer(Basicblock, nStages[3], n, stride=2, dropRate=dropRate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(nStages[3], num_classes)
        self.nStages = nStages[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _wide_layer(self, block, out_channels, num_blocks, stride, dropRate):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropRate))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.nStages)
        return self.fc(x)
```

## 7. Experimental Results

<img src="https://github.com/user-attachments/assets/0dc9c19c-c092-4c55-b1b0-10c2e5c0ee14" width="600" height="300">

- 동일한 Depth에서는 K가 클수록 우수
- 동일한 K에서는 Depth가 클수록 우수
- 동일한 파라미터 수에서는 Depth와 K가 제각각이라 실험에 신중할 필요가 있다.

### **Augmentation**

Random Horizontal Flip

Random Crop

Normalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

### Hyper Parmeters

Lerarning Rate = 0.1

SGD optimizer (momentum=0.9, weight decay=0.0005)

60번째, 120번째 에포크에서 학습률 0.1배 씩 감소

---

### Experiment 1

### WRN-28-10 vs WRN-40-1

**WRN-28-10 모델**

<img src="https://github.com/user-attachments/assets/4605eb56-91d9-4084-bb5d-0236b1637c00" width="700" height="200">

<img src="https://github.com/user-attachments/assets/fa1cc32a-af02-4ff1-9e4f-aa96b0485b9e" width="700" height="400">


Minimum Test Error Rate: 4.27%

**WRN-40-1 모델**

<img src="https://github.com/user-attachments/assets/990638c4-79d9-422f-846a-532832a737a4" width="700" height="200">

<img src="https://github.com/user-attachments/assets/c3d58808-1b78-49e4-8369-90e1e20a82c1" width="700" height="400">


Minimum Test Error Rate: 6.89%

### Experiment 2

### WRN-40-1(Drop out 적용) vs WRN-40-1(Drop out 적용 x)

**WRN-40-1(Drop out 적용)**

<img src="https://github.com/user-attachments/assets/c3d58808-1b78-49e4-8369-90e1e20a82c1" width="700" height="400">

Minimum Test Error Rate: 6.89%

<img src="https://github.com/user-attachments/assets/88dd4758-4b44-4355-a4b3-ded68b6c80b4" width="700" height="300">

**WRN-40-1(Drop out 적용 X)**

<img src="https://github.com/user-attachments/assets/2cb6cc87-2162-40ba-ad24-44857ea516bc" width="700" height="400">

Minimum Test Error Rate: 6.65%

<img src="https://github.com/user-attachments/assets/b7d9917c-99d8-4f12-a8fe-b6a1afe565a2" width="700" height="300">
