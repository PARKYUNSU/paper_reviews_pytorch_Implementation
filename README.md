# DeepLabV1

Dilation은 convolutional layer에 있는 파라미터로

케라스에서는 dilation rate로 파이토치에서는 dilation으로 입력을 받는다.

​

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
