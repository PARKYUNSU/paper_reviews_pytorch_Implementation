<details> <summary>**Siamese Networks 개요 및 설명 보기/접기**</summary>
Siamese Networks의 개요
Siamese Networks는 동일한 parameters나 weights을 공유하는 twin networks로 구성됩니다. 이 네트워크는 한 쌍의 inputs를 받아 각각의 features를 추출한 뒤 두 inputs 간의 유사도를 계산합니다. 이 유사도를 기반으로 분류 문제를 해결하며, 같은 클래스의 데이터는 거리를 최소화하고, 다른 클래스의 데이터는 거리를 늘리는 방식으로 학습됩니다.

Loss Functions
1. Contrastive Loss
Contrastive Loss는 이미지 pairs 사이의 차이를 학습시키기 위한 Loss입니다.

공식:

𝐿
=
𝑌
⋅
𝐷
2
+
(
1
−
𝑌
)
⋅
max
(
𝑚
𝑎
𝑟
𝑔
𝑖
𝑛
−
𝐷
,
0
)
2
L=Y⋅D 
2
 +(1−Y)⋅max(margin−D,0) 
2
 
𝐷
D: 이미지 features 사이의 거리
𝑚
𝑎
𝑟
𝑔
𝑖
𝑛
margin: 다른 클래스 간의 최소 거리 기준
특징:

같은 클래스의 샘플은 거리 
𝐷
D를 최소화
다른 클래스의 샘플은 
𝑚
𝑎
𝑟
𝑔
𝑖
𝑛
margin 이상으로 거리를 벌림
2. Triplet Loss
Triplet Loss는 anchor, positive, negative로 이루어진 triplet을 사용하여 anchor-positive 샘플의 거리를 최소화하고 anchor-negative 샘플의 거리를 최대화합니다.

공식:

𝐿
=
max
(
𝑑
(
𝑎
,
𝑛
)
−
𝑑
(
𝑎
,
𝑝
)
+
𝑚
𝑎
𝑟
𝑔
𝑖
𝑛
,
0
)
L=max(d(a,n)−d(a,p)+margin,0)
𝑑
(
𝑎
,
𝑝
)
d(a,p): anchor-positive 거리
𝑑
(
𝑎
,
𝑛
)
d(a,n): anchor-negative 거리
𝑚
𝑎
𝑟
𝑔
𝑖
𝑛
margin: 거리 기준
특징:

Positive 샘플은 anchor와 같은 클래스
Negative 샘플은 anchor와 다른 클래스
Pros and Cons
장점
각 클래스의 데이터 개수가 적어도 학습이 가능
불균형한 데이터로도 학습 가능
단점
데이터 pair 생성으로 인해 training 데이터 수가 많아질 수 있음
특정 task에 적합한 모델이 다른 task에 일반화하기 어려움
Input 데이터의 변형에 민감함
</details>
