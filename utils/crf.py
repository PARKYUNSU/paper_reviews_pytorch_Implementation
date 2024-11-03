import torch
import torch.nn.functional as F

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        """
        DenseCRF 클래스 초기화.
        
        Args:
        iter_max (int): CRF 추론 반복 횟수
        pos_w (float): 가우시안 페어와이즈 항 가중치
        pos_xy_std (float): 가우시안 필터에서 위치 차이 표준 편차 (spatial distance standard deviation)
        bi_w (float): 양방향 필터 가중치
        bi_xy_std (float): 양방향 필터에서 위치 차이 표준 편차
        bi_rgb_std (float): 양방향 필터에서 색상 차이 표준 편차
        """        
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        """
        CRF 추론 수행. 입력 이미지와 모델의 소프트맥스 확률 맵을 사용해 픽셀 단위 클래스 확률을 계산.
        
        Args:
        image (np.ndarray): 원본 이미지 (H, W, 3)
        probmap (np.ndarray): 모델이 출력한 소프트맥스 확률 맵 (C, H, W) - 클래스별 확률

        Returns:
        Q (np.ndarray): CRF 후처리된 확률 맵 (C, H, W)
        """
        C, H, W = probmap.shape
        image = image.permute((1, 2, 0))
        probmap = probmap.cpu().numpy()

        # U : Unray Energy
        U = crf_utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U) # memory 레이아웃을 연속적인 배열로 변환 (CRF에 맞게 사용)

        # 입력 이미지를 연속된 배열로 변환
        image = np.ascontiguousarray(image)

        # d : DenseCRF2D 객체 초기화 (이미지 크기 및 클래스 수 지정)
        d = dcrf.DenseCRF2D(W, H, C)
        
        # Unary Term 설정
        d.setUnaryEnergy(U)
        
        # Pairwise Term 추가: 가우시안 커널을 사용하여 공간적 관계를 고려
        # 공간적으로 가까운 픽셀들이 비슷한 클래스로 분류되도록 유도
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        
        # Pairwise Term 추가: 양방향 필터 사용 (공간적 거리 및 색상 정보를 동시에 고려)
        # 공간적으로 가깝고 색상이 비슷한 픽셀들이 같은 클래스로 분류되도록 유도        
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, # 거리 차이의 표준편차
            srgb=self.bi_rgb_std, # 색깔 차이의 표준편차
            rgbim=image, # 원본이미지
            compat=self.bi_w # 양방향 필터 가중치
        )
        # Q : 확률 분포 (최종 CRF 추론 결과)
        # 픽셀마다 각 클래스에 대한 확률 분포)
        Q = d.inference(self.iter_max) # 추론 반복 횟수
        Q = np.array(Q).reshape((C, H, W)) # C H W 재배열

        return Q