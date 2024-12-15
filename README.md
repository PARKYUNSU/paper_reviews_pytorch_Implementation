# SimSiam

"Exploring Simple Siamese Representation Learning" - 2020

-Xinlei Chen, Kaiming He (Facebook AI Research)

https://arxiv.org/pdf/2011.10566

---

## 1. Introduction

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



Self-Supervised Learning은 주로 Siamese Networks 구조를 사용합니다. Siamese Networks은 weight를 서로 공유하는 Twin Networks 구조를 가지고 있는데, 이들은 각 entitiy를 비교하는 데에 유용하게 사용할 수 있습니다.

그러나 Siamese Networks는 Collapsing(모든 Output이 일정한 값으로 수렴하는 현상)이 발생하는 문제가 있습니다. 이 현상을 해결하기 위해 기존에 다음의 연구들이 진행되었습니다.



## 2. Related Work
| **Method**          | **Approach**                                                                 | **Key Component**                                                                 |
|----------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Contrastive Learning** | SimCLR: Positive Pair는 끌어당기고 Negative Pair는 밀어내도록 학습                  | Negative Pairs로 Constant Output이 Solution Space에 포함되지 않도록 방지          |
| **Clustering**       | SwAV: Siamese Networks에 Online Clustering을 도입                             | Online Clustering으로 Collapsing 방지                                            |
| **BYOL**             | Positive Pairs만 사용                                                      | Momentum Encoder를 사용하여 Collapsing 방지                                      |

<img src="https://github.com/user-attachments/assets/fa34ddd5-4d2b-443a-9073-240b45fa3ae9" width=400>

| Comparison on Siamese architectures

### How SimSiam Emerges

SimSiam은 기존 방법론에서 Key Component 제거하여 더 간결한 구조를 갖게 되었습니다

### SimSiam: Simplified by Removing Key Components
| **Method**       | **Key Component Removed**         | **Result Model**                  |
|-------------------|-----------------------------------|---------------------------------------|
| **SimCLR**       | ❌ Negative Pairs                 | SimCLR without Negative Pairs      |
| **SwAV**         | ❌ Online Clustering             | SwAV without Online Clustering     |
| **BYOL**         | ❌ Momentum Encoder              | BYOL without Momentum Encoder      |
| **SimSiam**      | ➕ Adds Stop-Gradient              | SimSiam with Stop-Gradient         |

## 3. Method

<img src="https://github.com/user-attachments/assets/8541e2ab-40ca-42e3-8e98-3d5ec6a6683a" width=400>

| SimSiam Architecture

SimSiam은 자가 지도 학습에서 Collapsing 현상(출력이 일정한 값으로 수렴)을 방지하기 위해 설계된 단순한 구조입니다. Stop-gradient라는 독특한 기법을 활용하여 Negative Pair 없이도 학습을 안정적으로 수행할 수 있습니다.

### 3.1 Stop-Gradient

1. Collapsing 방지
  
   - 모든 출력이 일정한 값으로 수렴하면 모델이 더 이상 의미 있는 표현을 학습하지 못합니다.

   - Stop-gradient는 하나의 네트워크 출력($z1$ 또는 $z2$)을 상수처럼 취급하여 역전파를 차단

2. Gradient Flow 차단

   - $stopgrad(z) = z$ ($z$를 상수로 취급하여 역전파시 Gradient 계산하지 않도록)

   - forward 시 값을 그대로 사용, backward에서는 $\frac{∂stopgrad(z)}{∂z} = 0$

3. Gaussian 분포 유지

   - Stop-gradient는 $z$ 값이 Unit Hypersphere 위에 고르게 분포하도록 유도합니다.
   
   - 단위벡터 $\frac\{z}{||p_1\||_2}$ 의 채널별 표준편차(std)를 계산하여 Collapsing 여부 평가

---

### 3.2 SimSiam Architecture

1. Input Image $x$ → 랜덤 augmentation으로 $x_1, x_2$ 생성
2. 두 augmentation $x_1, x_2$ → 동일한 인코더 $f$ 통과
3. $f:$ 백본 + Projection MLP
4. $h:$ predictor → 한쪽에만 적용
5. Stop-gradient → 다른 한쪽에 적용

---

### 3.3 Loss
### 3.3.1 Negative Cosine Similarity

$D(p_1, z_2) = - \frac{p_1}{\||p_1\||_2} \cdot \frac{z_2}{\||z_2\||_2}$

$Where:$

- $p_1$: Predictor MLP의 출력 벡터
- $z_2$: Projection MLP의 출력 벡터 (stop-gradient 적용)
- $\||p_1\||_2$: $p_1$ 벡터의 $\ell_2$-노름(norm)
- $\||z_2\||_2$: $z_2$ 벡터의 $\ell_2$-노름(norm)

### 3.3.2 Symmetrized Loss

Symmetrized Loss는 두개의 Augmentation의 손실을 대칭적으로 계산하여 학습 안정성을 부여

$L = \frac{1}{2} D(p_1, stopgrad(z_2)) + \frac{1}{2} D(p_2, stopgrad(z_1))$

$Where:$

- $D(p_1,z_2)$: 두 벡터 $p_1$, $z_2$ 간의 Negative Cosine Similarity
- $p_1, p_2$: Predictor MLP의 출력 벡터
- $z_1, z_2$: Projection MLP의 출력 벡터
- $stopgrad(z)$: Stop-gradient 연산이 적용된 z, 상수로 취급 되어 Gradient 전파되지 않는 텐서

---

### 3.4 SimSiam 동작 원리

1. 하나의 Input image $x$에 대해 random augmentation으로 augmentation $x_1$, $x_2$ 생성

2. augmentation $x_1$, $x_2$는 Encoder $f$를 통과 (이떄, 두 Encoder는 weight을 공유)

3. Encoder를 통과한 두 Vectore 중 한쪽에만 Predictor $h$를 통과해 새로운 vector $p$를 만든다.

   $p_1 = h(f(x_1))$

   $z_2 = f(x_2)$

4. Symmetrized Loss
   - augmenatation $x1$에서 나온 $p_1$과 $z_2$간 손실 계산
   - augmenatation $x2$에서 나온 $p_2$과 $z_1$간 손실 계산
     (두 손실을 합산하고 평균을 내서 최종 손실로 사용)
5. Stop-gradient
   - 첫 번째 항에서는 $z_2$를 상수로 취급하여 gradient $z_2$로 전달되지 않음
   - 두 번째 항에서는 $z_1$에 stop-gradient가 적용
   - 두 augmentation이 학습 과정에서 균형을 이루도록 함
6. Loss Symmetry
   - 두 augmentation이 독립적으로 학습에 기여하므로 한쪽 네트워크가 과도하게 학습되지 않도록 방지
   - 모델이 양쪽 입력에 대해 균형 잡힌 표현을 학습할 수 있도록 도움

### 3.4.1 시나리오

### 1. **이미지 A와 이미지 B가 들어가는 경우**

#### **Step 1. Augmentation 적용**

- 이미지 A와 B 각각에 대해 서로 다른 랜덤 변형(augmentation)을 적용하여 두 개의 뷰를 생성합니다.

  예를 들어:
  - $x_a^{1}$: 이미지 A에 augmentation 1 적용
  - $x_a^{2}$: 이미지 A에 augmentation 2 적용
  - $x_b^{1}$: 이미지 B에 augmentation 1 적용
  - $x_b^{2}$: 이미지 B에 augmentation 2 적용

#### **Step 2. Encoder를 통해 Feature 추출**

- 각각의 augmentation 이미지를 Encoder $f$에 입력하여 Feature Vector $z$를 생성합니다.
  - $z_a^{1} = f(x_a^{1})$
  - $z_a^{2} = f(x_a^{2})$
  - $z_b^{1} = f(x_b^{1})$
  - $z_b^{2} = f(x_b^{2})$

#### **Step 3. Projection MLP를 통과**

- 추출된 Feature Vector $z$를 Projection MLP에 입력하여 고차원 표현 공간으로 매핑합니다.
  - $z_a^{1}, z_a^{2} \rightarrow \text{Projection MLP} \rightarrow z_a'^{1}, z_a'^{2}$
  - $z_b^{1}, z_b^{2} \rightarrow \text{Projection MLP} \rightarrow z_b'^{1}, z_b'^{2}$

#### **Step 4. Predictor를 통해 정렬**

- Projection MLP의 출력을 Predictor $h$에 입력하여 다른 뷰와 정렬하도록 학습합니다.
  - $p_a^{1} = h(z_a'^{1}), p_a^{2} = h(z_a'^{2})$
  - $p_b^{1} = h(z_b'^{1}), p_b^{2} = h(z_b'^{2})$

#### **Step 5. Symmetrized Loss 계산**

- 이미지 A와 B 각각의 augmentation 쌍에 대해 대칭 손실을 계산합니다.
  - 이미지 A
  
    $L_A = \frac{1}{2} D(p_a^{1}, \text{stopgrad}(z_a'^{2})) + \frac{1}{2} D(p_a^{2}, \text{stopgrad}(z_a'^{1}))$
    
  - 이미지 B
    
    $L_B = \frac{1}{2} D(p_b^{1}, \text{stopgrad}(z_b'^{2})) + \frac{1}{2} D(p_b^{2}, \text{stopgrad}(z_b'^{1}))$

#### **Step 6. 가중치 업데이트**

- $L_A + L_B$의 총 손실을 기반으로 모델 가중치를 업데이트합니다.

- 전체 과정은 positive pair 간의 관계를 학습하는 데 집중하며, 이미지의 개수와 관계없이 병렬 처리가 가능

---

### 2. **이미지 C가 추가로 들어가는 경우**

이미지 C가 추가로 들어오면, 동일한 과정을 반복하지만 손실 계산은 A, B, C 각각의 augmentation에 대해 독립적으로 수행됩니다.

#### **Step 1 - 4: 동일한 과정 반복**

- $x_c^{1}, x_c^{2}$ 에 augmentation 적용
- Encoder와 Projection MLP, Predictor를 통해 $p_c^{1}, p_c^{2}$ 생성

#### **Step 5. Symmetrized Loss 계산 (C 포함)**

이미지 A, B, C 각각에 대해 개별적으로 Loss를 계산:
- 이미지 A
  
  $L_A = \frac{1}{2} D(p_a^{1}, \text{stopgrad}(z_a'^{2})) + \frac{1}{2} D(p_a^{2}, \text{stopgrad}(z_a'^{1}))$
  
- 이미지 B
  
  $L_B = \frac{1}{2} D(p_b^{1}, \text{stopgrad}(z_b'^{2})) + \frac{1}{2} D(p_b^{2}, \text{stopgrad}(z_b'^{1}))$

- 이미지 C
  
  $L_C = \frac{1}{2} D(p_c^{1}, \text{stopgrad}(z_c'^{2})) + \frac{1}{2} D(p_c^{2}, \text{stopgrad}(z_c'^{1}))$

#### **Step 6. 총 손실 계산**

- 최종 손실
  
  $L_{\text{total}} = L_A + L_B + L_C$

#### **Step 7. 모델 업데이트**

- 총 손실 $L_{\text{total}}$ 을 기반으로 모델 가중치를 업데이트

- 전체 과정은 positive pair 간의 관계를 학습하는 데 집중하며, 이미지의 개수와 관계없이 병렬 처리가 가능


---

## 4. Empirical Study
### 4.1 Stop-Gradient의 역할
   -  Stop-gradient가 없으면 Collapsing 문제가 발생
   -  Stop-graident는 Collapsing을 방지하며, $z$값의 분산이 유지
   -  단위벡터 $\frac\{z}{||p_1\||_2}$ 의 채널별 표준편차(std)를 계산하여 Collapsing 여부 평가

1) Collapsing 발생 시
   
   $z$가 상수로 수렴하며, 채널별 표준편차가 0에 가까워집니다.

3) Gaussian 분포에 따른 경우
   
   $z$가 Unit Hypersphere 위에 고르게 분포되며, 표준편차는 특정 값에 근접합니다.

   $std ≈ \frac{1}{\sqrt{d}}$  ($d$는 $z$ 벡터의 차원)

---

### 4.2 Predictor

- **Predictor의 역할**:  
  - Predictor $h$는 Projection MLP의 출력 $z$를 다른 augmentation의 $z$와 정렬되도록 학습 
  - 표현을 최신 상태로 유지하며, augmentation 분포의 기대값을 근사하는 데 도움

- **실험 결과**:  
  - **Predictor 제거**:  
    - $h$를 제거하면 모델이 collapsing(출력이 상수로 수렴)하여 학습 불가
  - **고정된 Predictor**:  
    - 무작위 초기화 상태로 고정된 $h$는 학습하지 못하며, 손실이 높음
  - **학습 가능한 Predictor**:  
    - 동적으로 업데이트되는 $h$는 SimSiam의 안정적 학습을 도움

---

### 4.3 Batch Size

- **효과**:  
  - SimSiam은 다양한 배치 크기(64~4096)에서 안정적으로 작동
  - 배치 크기가 작아도 성능 저하가 없음

- **SimCLR 및 SwAV와의 차이점**:  
  - SimCLR와 SwAV는 큰 배치 크기(4096 이상)가 필요하지만, SimSiam은 일반적인 크기(512)에서도 작동

---

### 4.4 Batch Normalization

- **효과**:  
  - Batch Normalization(BN)은 각 레이어의 출력 분포를 안정화하여 최적화
  - SimSiam에서 BN은 collapsing 방지에는 직접적인 영향을 미치지 않음  

- **구성**:  
  - **Projection MLP**:  
    - BN은 모든 FC 레이어에 적용되며, 출력 FC에는 ReLU가 없음
  - **Prediction MLP**:  
    - BN은 은닉 FC 레이어에 적용되지만, 출력 FC에는 적용되지 않음

---

### 4.5 Symmetrization

- **수식**:  
  $$L = \frac{1}{2} D(p_1, stopgrad(z_2)) + \frac{1}{2} D(p_2, stopgrad(z_1))$$  
  - 두 augmentation 간 손실을 대칭적으로 계산하여 학습 안정성을 높임

- **효과**:  
  - 대칭화는 정확도를 높이는 데 기여하지만, collapsing 방지에는 필수적이지 않음

- **비교 실험**:  
  - 대칭화 제거(비대칭 손실)를 적용해도 모델이 collapsing 없이 학습할 수 있지만, 정확도는 약간 감소
