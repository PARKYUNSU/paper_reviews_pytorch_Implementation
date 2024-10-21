# DeepLabV1

Dilation은 convolutional layer에 있는 파라미터로

케라스에서는 dilation rate로 파이토치에서는 dilation으로 입력을 받는다.

​

dilation을 한 마디로 말하자면 convolution 커널의 간격을 의미한다.

​![alt text](image.png)
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
