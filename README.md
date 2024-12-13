# SimSiam

"Exploring Simple Siamese Representation Learning" - 2020

-Xinlei Chen, Kaiming He (Facebook AI Research)

---

## Introduction

### Siamese Networks

<details> <summary>Siamese Networks 보기</summary>
 
 DeepLearning에서는 학습을 위해 많은 양의 데이터를 필요로 합니다. 그래서 데이터가 부족하다는 말은, DeepLearning 모델의 성능이 좋지 않음을 암시합니다.

 그래서 고안된 Siamese Networks는 데이터 양이 적거나, Imbalanced Class Distribution한 데이터에서도 모델의 정확성을 높힐 수 있습니다.
 
 Siamese Networks는 동일한 parameters나 weights을 공유하는 Twin Networks로 구성됩니다.
 
 이 네트워크는 한 쌍의 inputs를 받아 각각의 features를 추출한 뒤 두 inputs 간의 유사도를 계산합니다. 이 유사도를 기반으로 Classification을 수행하며, 같은 Class의 데이터는 거리를 최소화하고, 다른 Class의 데이터는 거리를 늘리는 방식으로 학습됩니다.

```python
# Twin Network
class TwinNetwork(nn.Module):
    def __init__(self):
        super(TwinNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))

    def forward(self, x):
        return self.shared_layers(x)

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.twin_network = TwinNetwork()

    def forward(self, input1, input2):
        output1 = self.twin_network(input1)
        output2 = self.twin_network(input2)
        return output1, output2
```

- Siamese Networks 장점

  각 클래스의 데이터 개수가 적어도 학습이 가능

  불균형한 데이터로도 학습 가능

- Siamese Networks 단점

  데이터 pair 생성으로 인해 training 데이터 수가 많아질 수 있음

  특정 task에 적합한 모델이 다른 task에 일반화하기 어려움

  Input 데이터의 변형에 민감함


<img src="https://github.com/user-attachments/assets/d54df74e-8f8c-4d23-9985-a40c948c2ee7" width=500>

<img src="https://github.com/user-attachments/assets/e91bfd4f-1db7-4727-8c60-1b84a3b2a66f" width=300>


Loss Functions

1. Contrastive Loss

   Contrastive Loss는 이미지 pairs 사이의 차이를 학습시키기 위한 Loss

   $𝐿=𝑌⋅𝐷^2+(1−𝑌)⋅max(𝑚𝑎𝑟𝑔𝑖𝑛−𝐷,0)^2$

   $Where:$
   
   - $D:$ 이미지 features 사이의 거리

   - $margin:$ 다른 클래스 간의 최소 거리 기준

   - 𝑌: 두 샘플의 관계를 나타내는 레이블 ($Y = 1$ : 같은 클래스, $Y = 0$ : 다른 클래스)

   특징:
   
   - 같은 Class의 샘플: 거리 D를 최소화
      
   - 다른 Class의 샘플: 거리를 margin 이상으로 벌림
  
<img src="https://github.com/user-attachments/assets/5264586d-d74a-44fe-9f17-f4565b30b215" width=500>


2. Triplet Loss
   
   Triplet Loss는 Anchor, Positive, Negative로 이루어진 triplet을 사용하여, Anchor와 Positive 거리를 최소화, Anchor와 Negative 거리를 최대화하게 학습

   $L=max(d(a,n)−d(a,p)+margin,0)$

   $Where:$
   
   - $d(a,p):$ Anchor-Positive 거리

   - $d(a,n):$ Anchor-Negative 거리

   - $margin:$ 거리 기준

   - $Anchor:$ Triplet Loss에서 참조 역할을 하는 Input Sample

   - $Positive:$ Anchor와 같은 클래스에 속하는 데이터.
  
   - $Negative:$ Anchor와 다른 클래스에 속하는 데이터.

<img src="https://github.com/user-attachments/assets/5e0ac3d3-52d0-41af-9ccf-6862098b913d" width=500>
     
     Anchor: "A"라는 데이터
     
     Positive: 다른 "A"의 데이터 (같은 클래스)
     
     Negative: "B"라는 데이터 (다른 클래스)

</details>




