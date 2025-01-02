# LSTM

"LONG SHORT-TERM MEMORY" - 1997

-Sepp Hochreiter, Jürgen Schmidhuber

https://www.bioinf.jku.at/publications/older/2604.pdf

---

## 순환 신경망 (Recurrent Neural Network, RNN)

순환 신경망(RNN)은 순서가 있는 연속적인 데이터를 처리하는 데 적합한 인공 신경망입니다. RNN은 각 시점(time step)의 데이터가 이전 시점의 데이터와 상호작용하며, 이러한 특성으로 인해 **시간적 연속성**이 있는 데이터(예: 자연어, 시계열 데이터, 음성 데이터)를 효과적으로 처리할 수 있습니다.

---

### 연속형 데이터의 특징

연속형 데이터란 각 시점의 데이터가 이전 시점의 데이터와 독립적이지 않고, 강한 상관관계를 가지는 데이터를 말합니다.

- **자연어**: 단어는 이전에 등장한 단어의 영향을 받아 문장의 의미를 형성합니다.  
  예) "금요일이 지나면" → "주말" 또는 "토요일" 등의 단어 예상 가능.

- **주식 데이터**: 특정 날짜의 주가는 이전 날짜의 주가 및 추세에 영향을 받습니다.

- **음성 데이터**: 발음의 연속성은 앞뒤의 음소 간 관계를 포함합니다.

---

### RNN의 특징

순환 신경망은 연속형 데이터를 순차적으로 처리하여 각 시점마다 **은닉 상태(hidden state)**를 계산하고 저장합니다.  
RNN의 핵심은 이전 시점의 정보를 은닉 상태로 유지하며, 이를 현재 입력과 결합하여 다음 시점의 출력을 계산하는 구조입니다.

#### 동작 원리
1. **입력값**과 **이전 시점의 은닉 상태**를 이용해 현재 은닉 상태를 계산합니다.
2. 은닉 상태는 현재 입력값, 가중치, 편향을 기반으로 활성화 함수(예: tanh)를 통해 계산됩니다.
3. 은닉 상태를 기반으로 최종 출력값을 도출합니다.

---

### RNN 수식

RNN의 은닉 상태와 출력값 계산 공식은 다음과 같습니다:

#### 은닉 상태 계산

$h_t = σ(W_h h_{t-1} + W_x x_t + b)$

- $h_t:$ 현재 시점의 은닉 상태  
- $h_{t-1}:$ 이전 시점의 은닉 상태  
- $x_t:$ 현재 입력값  
- $W_h, W_x:$ 가중치  
- $b:$ 편향  
- $σ:$ 활성화 함수 (예: tanh)

---

### 구조 설명

<img src="https://github.com/user-attachments/assets/98486a20-a16f-458a-bc79-0686d311d4cb" width=500 alt="RNN 도식화">

| RNN 도식화

1. **입력 벡터 ($x_t$)**: Input Layer로 들어오는 데이터.  
2. **출력 벡터 ($y_t$)**: Output Layer에서 나오는 결과값.  
3. **메모리 셀 (Memory Cell)**: Hidden Layer에서 활성화 함수를 통해 값을 계산하고, 이전 시점의 출력값을 기억합니다. 이를 **RNN Cell**이라고도 부릅니다.  

#### 메모리 셀의 역할
메모리 셀이 값을 "기억한다"는 것은 이전 시점($t-1$)의 Hidden State ($h_{t-1}$)를 현재 시점의 Hidden State ($h_t$) 계산에 입력으로 사용하는 것을 의미합니다.

---

<img src="https://github.com/user-attachments/assets/0da3970a-20e3-4f32-96e6-a73f07a0dfb9" width=500 alt="RNN 표현방식">

| 재귀적 표현(좌측), 펼친 표현(우측)

---

### Hidden State의 계산
- **Hidden State ($h_t$)** 는 현재 입력값 ($x_t$)과 이전 Hidden State ($h_{t-1}$)를 조합하여 계산됩니다.  
- 이를 통해 RNN은 과거 정보를 유지하고 활용하며, 연속적인 데이터의 시간적 패턴을 학습합니다.

---

## RNN의 문제점

RNN은 연속적인 데이터를 처리하는 데 적합하지만 몇 가지 한계점이 존재합니다

1. **기울기 소실(Vanishing Gradient)**  
   - RNN은 시간이 길어질수록 기울기가 점점 작아지는 현상이 발생합니다.  
   - 이로 인해 RNN은 긴 시계열 데이터에서 초반 시점의 정보를 학습하기 어려워집니다.

2. **기울기 폭발(Exploding Gradient)**  
   - 반대로, 기울기가 너무 커져서 학습 과정에서 수치적으로 불안정해지는 경우도 있습니다.

3. **장기 의존성 문제(Long-term Dependency)**  
   - RNN은 시간적으로 먼 과거의 정보를 효율적으로 기억하지 못합니다.  
   - 이는 긴 문장이나 긴 시계열 데이터를 학습할 때 주요한 한계점으로 작용합니다.

---

## LSTM

RNN의 한계점을 극복하기 위해 **LSTM(Long Short-Term Memory)** 이 제안되었습니다.  
LSTM은 기존 RNN의 구조에 **셀 상태(Cell State)** 와 **게이트 구조** 를 추가하여 긴 시간 간격의 정보를 효과적으로 학습할 수 있습니다.

<img src="https://github.com/user-attachments/assets/c40ee410-ab99-46fc-9f53-24f77aa8cf49" width=500>

| RNN to LSTM

### LSTM의 주요 특징
1. **Cell State ($C_t$)**
- 수평으로 흐르는 주요 경로로, 정보를 장기간 유지하거나 제거할 수 있는 컨베이어 벨트 역할을 합니다.
- 정보가 거의 변경되지 않은 채로 전달될 수 있도록 설계되었습니다.

2. **Gate**
- 정보를 선택적으로 추가하거나 제거하는 역할을 합니다.
- 각 게이트는 Sigmoid Layer와 Pointwise 곱셈으로 구성됩니다.
  - Sigmoid는 0과 1 사이의 값을 반환하며, 정보의 중요도를 결정합니다.
  - 값이 0이면 정보를 전달하지 않고, 1이면 모든 정보를 전달합니다.

3. **게이트 구조(Gates)**  
- **입력 게이트(Input Gate)**: 새로운 정보를 얼마나 받아들일지 결정  
- **망각 게이트(Forget Gate)**: 이전 정보를 얼마나 버릴지 결정
- **출력 게이트(Output Gate)**: 다음 시점으로 보낼 정보를 결정

LSTM은 이러한 구조를 통해 RNN의 단점을 완화하고, 긴 시계열 데이터에서도 안정적인 학습 성능을 제공합니다.

---
## LSTM의 단계별 동작

**1. Forget Gate Layer (기존 정보의 제거)**
   
이 단계에서는 **이전 상태 ($h_{t−1}$)** 와 **현재 입력 ($x_t$)** 를 기반으로, **Cell State ($C_{t−1}$)** 에서 어떤 정보를 버릴지 결정합니다.

Sigmoid Layer를 통해 0과 1 사이의 값인 $f_t$를 계산하며, 값이 0에 가까울수록 해당 정보를 잊게 됩니다.

<img src="https://github.com/user-attachments/assets/716dc352-b9b4-4e62-bb29-451b64de77f2" width=500>

**2. Input Gate Layer (새로운 정보 추가)**
   
새로운 정보를 Cell State에 얼마나 추가할지 결정합니다.

이 단계는 두 부분으로 나뉩니다:

Input Gate ($i_t$): Sigmoid Layer가 어떤 값을 업데이트할지 결정.

Candidate Value ($C~_t$): Tanh Layer를 통해 새롭게 추가할 후보 값을 생성.

<img src="https://github.com/user-attachments/assets/b63407f1-bf32-4fbd-aa51-d2c84b0b99e8" width=500>

**3. Cell State 업데이트**
   
이전 단계에서 계산한 값을 이용해 **Cell State ($C_t$)** 를 업데이트합니다.

Forget Gate와 Input Gate의 결과를 조합하여 새 상태를 생성합니다.

이 과정은 이전 상태에서 잊어야 할 정보는 제거하고, 새로운 정보를 추가하는 방식으로 이루어집니다.

<img src="https://github.com/user-attachments/assets/0e9c2cff-d915-472e-9380-487327af97ce" width=500>

**4. Output Gate Layer (출력 생성)**
   
최종적으로 어떤 정보를 출력할지 결정합니다.

이 단계는 다음 두 단계로 이루어집니다:

Output Gate ($o_t$): Sigmoid Layer가 출력할 정보를 결정.

Filtered Output: Cell State를 Tanh Layer에 통과시켜 -1과 1 사이의 값으로 변환한 뒤, Output Gate와 곱하여 최종 출력값 생성.

<img src="https://github.com/user-attachments/assets/1660b37d-5d41-4b21-ad52-98e921074070" width=500>


## Experiment

### Experimental Setup
|                    | Details                          |
|--------------------|----------------------------------|
| Optimizer          | Adam                             |
| Loss Function      | CrossEntropyLoss                 |
| Number of Epochs   | 30                               |
| Batch Size         | 32                               |
| Learning Rate      | 0.001                            |
| Device             | kaggle, GPU P100                 |

---
### Data Description

|                    | Details                                                                |
|--------------------|------------------------------------------------------------------------|
| Dataset            | Sarcasm Dataset                                                       |
| Number of Samples  | Approximately 26,000                                                 |
| Number of Classes  | 2 (Sarcastic, Non-Sarcastic)                                          |
| Vocabulary Size    | 1000 (max tokens)                                                    |
| Minimum Word Frequency | 2                                                                |
| Maximum Sentence Length | 120 tokens                                                     |
| Train/Test Split   | 80:20                                                                 |


### Result

#### Best Epoch Details

| **Metric**            | **Epoch** | **Value**   |
|------------------------|-----------|-------------|
| Best Validation Loss   | 13        | **0.0119**  |
| Best Validation Accuracy | 24        | **0.8325**  |

---

#### Additional Observations

1. **Best Validation Loss**:
   - Occurred at **Epoch 13** with a value of **0.0119**
   - Corresponding Train Loss: **0.0099**
   - Corresponding Train Accuracy: **86.13% (0.8613)**
   - Validation Accuracy at this epoch: **82.83% (0.8283)**

2. **Best Validation Accuracy**:
   - Achieved at **Epoch 24** with a value of **83.25% (0.8325)**
   - Corresponding Validation Loss: **0.0128**
   - Corresponding Train Loss: **0.0084**
   - Corresponding Train Accuracy: **87.94% (0.8794)**

3. **Final Epoch (30)**:
   - Train Loss: **0.0077**, Train Accuracy: **88.95% (0.8895)**
   - Validation Loss: **0.0142**, Validation Accuracy: **82.55% (0.8255)**

<img src="https://github.com/user-attachments/assets/98da957e-f89a-44af-8442-bcf51580d769">

