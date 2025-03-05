# CRAG

"Corrective Retrieval Augmented Generation" - 2024

ㅡ Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling

[Read the Paper](https://arxiv.org/pdf/2401.15884)

---
# 1. Introduction

LLM의 대표적인 문제인 Hallucination(환각) 현상은 선행된 연구인 [RAG](https://github.com/PARKYUNSU/pytorch_imple/tree/main/Agentic_RAG/Basic_Agentic_RAG)에 의해 다소 보완되었으나, RAG의 주요 문제는 검색(Retrieval) 단계에 있습니다. 이는 검색된 문서의 연관성에 크게 의존하게되어, 질문(Query)에 도움이 되지 않는 부정확하거나 불필요한 정보가 포함될 수 있기 때문입니다.

이에 본 논문은, 부정확한 검색 결과를 줄이고 생성의 견고성을 향상시키기 위해서 Corrective Retrieval Augmented Generation (CRAG)을 제안합니다.

CRAG는 내부 문서의 관련성 score를 기반으로 confidence를 계산해서, 문서들을 Correct, Incorrect, Ambiguous 세가지로 구분한 후, 각 파트에 맞게끔 문서를 정제하거나 외부 지식을 보완하여 최종 답변을 제공해서 문제를 보완합니다.

# 2. Related Work
### 1. Hallucinations of LLMs
LLM은 명령어 이해를 바탕으로 텍스트를 생성하지만, 여전히 가장 심각한 문제 중 하나는 Hallucinations(환각) 문제입니다. 부정확한 지식 및 과거 정보를 기반으로 문장을 이해 및 답변하는 모델은 Fine-tuning 및 재학습을 통해 더 정확한 정보를 재공함으로써 환각 현상을 막을 수 있습니다. 그러나 이 방법은 시간과 비용이 많이 들 수밖에 없습니다.

### 2. RAG (Retrieval-Augmented Generation)
기존 RAG는 LLM의 입력된 Query에 검색된 문서를 추가적인 정보원으로 재공함으로써 환각 문제를 완화하는 모델입니다. 하지만 앞서 설명한 RAG의 문제는 그 자체 검색(Retrieval)에 문제가 있으며 부정확한 검색은 답변 결과에 영향을 끼치기 마련입니다.

<img src="https://github.com/user-attachments/assets/ab8c0c94-1fd8-4ab5-854f-8d5de42804fb" width=400>

RAG는 Retrieval $R$과 Generator $G$로 나뉩니다. 입력 $X$와 $C = {d_1, …, d_N}$로 이루어진 대량의 Document에서 상위 $K$ 개의 문서 $D = {d_{r1}, …, d_{rk}}$를 검색해서 답변 $Y$를 생성하는 프로세스입니다. 이 과정을 수식으로 표현하면,

$$P(Y|X) = P(D|X)P(Y, D|X)$$

Retrieval과 Generator는 서로 영향을 긴밀하게 주고 있음을 보여줍니다, 즉, 검색이 실패하면 생성자가 아무리 뛰어나도 제대로 된 답변을 할 수 없음을 볼 수 있습니다.

### 3. Advanced RAG
또한, 최근 연구에서는 검색된 문서가 항상 정답을 보장하지 않을 수 있기 때문에, 어떤 경우에는 LLM 스스로 Retrieval 없이 답변하는 것이 더 정확할 수도 있다고 봅니다. 그래서 이런 자가 결정을 돕기 위해 Self-RAG 같은 접근법에서는 Critic Model 즉 평가자 모델을 도입해서, 검색을 할지 아니면 LLM이 스스로 답변을 할지 판단합니다.

# 3. CRAG
CRAG는 기존 접근법들과 다르게, 검색기능을 답변을 생성하는 보조 도구로 활용하거나 검색의 필요 여부에 집중하는 것이 아닌, 검색기가 부정확한 결과를 검색하는 상황 자체를 중점적으로 다루면서 RAG의 고질적인 문제를 해결하고자 설계되었습니다.

### 3.1. Overview of Model Inference
CRAG의 모델은 Retrieval, Knowledge Correction, Generation으로 나뉩니다.

Retreieval에서는 입력된 Query에 맞게 문서를 검색하고, 경량화된 Retrieval Evaluator(평가자)로 Query와 검색된 문서의 관련성 Score를 추정합니다.
이 Score는 총 3가지의 Cofidence로 정량화되어, 1) Correct, 2) Incorrect, 3) Ambiguous 로 나뉘어서 동작합니다.

#### 1) Correct
검색된 문서들이 더 정밀한 Knowledge Strips로 정제되며, Decomposition, Filter, Recomposition 과정을 거쳐서 정제됩니다.

#### 2) Incorrect
검색된 문서들을 사용하지 않고, Web Search로 검색된 정보를 사용합니다.

#### 1) Ambiguous
검색된 문서가 부정확한 문서인지 아닌지 경정 내릴 수 없는 상태로, 이런 경우에는 두 가지 동작을 모두 사용하여 균형잡힌 검색 결과가 나오도록 유도합니다.

![image](https://github.com/user-attachments/assets/e080a3eb-971b-4310-9ca5-f3ed47cd6d6a)

### 3.2. Retrieval Evaluator

