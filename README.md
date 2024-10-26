# DeepLabV1
---
Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs

저자 : Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
---

1. Introduction
** Deep Convolutional Neural Networks ** (DCNN)은 컴퓨터 비전에서 큰 영향을 끼쳤습니다. 이로 인해 이미지 분류 성능이 크게 향상되었으나, 하지만 문제점이 있었습니다.

우리가 목표로 하는 Semantic Segmentation(의미론적 분할)에서는 DCNN만으로는 부족합니다. DCNN은 분류 작업에 최적화되어 있어 이를 활용하려면 몇 가지 추가적인 작업이 필요합니다. DCNN도 Semantic Segmentation 작업을 위해서는 추가적인 요소가 필요합니다.

그럼, DCNN을 활용해 어떻게 Semantic Segmentation을 구현할 수 있을까요? 고려해야 할 몇 가지 과제가 있습니다:

해상도 축소: 특징 맵(feature map)의 해상도가 점점 줄어들면서 자세한 부분을 놓칠 수 있습니다.
다중 스케일 객체 탐지: 서로 다른 크기의 객체를 인식하는 것이 어려워집니다.
정확도 감소: 다운샘플링 과정에서 위치 정보가 불확실해집니다.
1.1 해상도 축소 문제
DCNN에서 다운샘플링은 보통 stride와 max-pooling으로 이루어지며, 이로 인해 최종적으로 사용 가능한 공간 해상도가 원본 이미지보다 현저히 작아집니다. 그 결과로, 고해상도의 상세한 분할이 어려워집니다.

이를 해결하기 위해, 일부 max-pooling 계층을 제거하고 대신 후속 컨볼루션 계층에서 업샘플링 필터를 사용하여 더 높은 샘플링 해상도로 특징 맵을 생성할 수 있습니다. 이 역할을 하는 것이 바로 Atrous(확장) 필터로, 효율적으로 넓은 공간을 탐색할 수 있게 해줍니다.

1.2 다중 스케일 객체의 존재
DCNN에 이미지를 넣으면 여러 스케일의 필터들이 적용되며, 각 스케일의 특징이 추출됩니다. 하지만 이 과정에서 연산량이 크게 늘어나 속도 측면에서의 성능이 저하됩니다.

이를 해결하기 위해 제안된 방법이 Spatial Pyramid Pooling입니다. 다양한 유효 시야를 가진 필터들을 사용해 여러 스케일에서 객체와 유용한 맥락 정보를 포착합니다. 이때, 단순 리샘플링 대신 다중 병렬 Atrous 컨볼루션 계층을 사용하는 것이 효율적이며 이를 **Atrous Spatial Pyramid Pooling(ASPP)**이라 부릅니다.

1.3 정확도 감소 문제
Semantic Segmentation에서 픽셀 단위의 정밀한 예측이 필요하지만, DCNN을 기반으로 하면 다운샘플링 과정에서 정밀도가 떨어질 수 있습니다. 이를 보완하기 위해 **CRF(Conditional Random Fields)**가 활용됩니다. CRF는 각 픽셀의 클래스 점수를 주변 픽셀과의 상호작용을 통해 결합하여 더 정확한 예측을 가능하게 합니다.

2. Method
논문의 저자들은 3가지 방법을 제안했습니다. 이들 각각에 대해 더 깊이 살펴보겠습니다.

2.1 밀집 특징 추출을 위한 Atrous Convolution
Atrous Convolution 방식은 해상도를 줄이지 않으면서 특징을 추출할 수 있는 방법으로, 표준 CNN의 다운샘플링으로 인한 해상도 손실 문제를 완화합니다.

이때, rate parameter를 조정하여 픽셀 간의 간격을 두고 필터를 적용함으로써 연산량을 증가시키지 않으면서 해상도를 유지하는 Dense feature extraction이 가능합니다.

2.2 ASPP를 활용한 다중 스케일 이미지 표현
ASPP(Atrous Spatial Pyramid Pooling)는 다양한 rate 값의 여러 Atrous Convolution 계층을 병렬로 적용하여 여러 스케일에서 객체와 이미지를 탐지할 수 있도록 합니다.

이는 R-CNN의 Spatial Pyramid Pooling에서 영감을 얻은 방식으로, 각기 다른 스케일의 Atrous Convolution 계층이 병렬로 적용된 후 각각의 특징들이 결합됩니다.

2.3 정확한 경계 복구를 위한 CRF 적용
DCNN을 통해 얻게 된 특징 맵을 Bilinear Interpolation으로 원본 이미지 크기로 확대한 후, CRF를 사용해 픽셀 단위의 정밀도를 높입니다. CRF는 각 픽셀의 위치와 색상을 기준으로 주변 픽셀과의 상호작용을 통해 비슷한 컬러를 가진 인접 픽셀에 동일한 레이블을 부여함으로써 더욱 정확한 경계 복구를 가능하게 합니다.

Dilation은 convolutional layer에 있는 파라미터로

케라스에서는 dilation rate로 파이토치에서는 dilation으로 입력을 받는다.



dilation을 한 마디로 말하자면 convolution 커널의 간격을 의미한다.

![image](https://github.com/user-attachments/assets/b1a3b425-8910-4450-94c2-2b1047746fd0)
출처: https://www.semanticscholar.org/paper/Deep-Dilated-Convolution-on-Multimodality-Time-for-Xi-Hou/afadf82529110fadcbbe82671d35a83f334ca242

​

dilation이 2라면 커널 사이의 간격이 2가 되는 것이고, 커널의 크기가 (3,3)이라면 (5,5) 커널과 동일한 넓이가 되는 것이다.

​

필터 내부에 zero padding을 추가해 강제로 receptive field를 늘린다.

즉, weight가 있는 부분을 제외한 나머지 부분은 전부 0으로 채워진다.

​

기존의 receptive field의 한계를 극복하고 좀 더 넓은 간격의 커널을 적은 리소스로 사용할 때 많이 활용된다.

​

이 파라미터는 wavenet 같은 유명한 신경망을 구현할 때도 많이 활용된다.

1. Mixed Precision Training Ole!?
   대부분의 deep learning framework(eg. PyTorch, TensorFlow)들은 모델을 training할 때 float32(FP32) data type을 사 용하게 됩니다. 즉, 모델의 weight와 input data가 모두 FP32(32bit)의 data type을 가진다는 뜻입니다. 이와 다르게 Mixed-p recision training은 single-precision(FP32)와 half-precision(FP16) format을 결합하여 사용하여 모델을 training하 는 방식입니다.
   (FP16 data type은 FP32와 다르게 16bit만을 사용하게 됩니다.)
   Mixed-precision training방식을 통해 다음과 같은 장점을 가집니다.
   • FP32로만 training한 경우와 같은 accuracy 성능을 도출
   • Training time이 줄음
   • Memory 사용량이 줄음
   • 이로 인해 더 큰 batch size, model, input을 사용 가능하게 함
   Mixed-precision training은 NVIDIA에 의해 처음 제안되었고 Automatic Mixed Precision(AMP)라는 feature를 개발하였습 니다. AMP feature는 특정 GPU operation을 FP32에서 mived precision으로 자동으로 바꿔주었으며 이는 performance를 향상시키면서 accuracy는 유지하는 효과를 가져왔습니다.
