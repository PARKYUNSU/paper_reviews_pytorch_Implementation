# CRAG

"Corrective Retrieval Augmented Generation" - 2024

ㅡ Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling

[Read the Paper](https://arxiv.org/pdf/2401.15884)

---
# Introduction

LLM의 대표적인 문제인 Hallucination(환각) 현상은 선행된 연구인 [RAG](https://github.com/PARKYUNSU/pytorch_imple/tree/main/Agentic_RAG/Basic_Agentic_RAG)로 많이 보안이 되었습니다.

그러나, RAG의 가장 문제는 Retrieval입니다. 이는 검색된 문서의 연관성에 크게 의존하게되어, 검색된 문장에 질문(Query)에 도움이 되지 않는 문서가 섞일 수 있기 때문입니다.

부정확한 검색 결과를 줄이고 생성의 견고성을 향상시키기 위해서 본 논문에서는 Corrective Retrieval Augmented Generation (CRAG)을 제안합니다.

CRAG는


