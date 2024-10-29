import numpy as np
import cv2

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        """
        Args:
        iter_max (int): CRF-like 추론 반복 횟수
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
        CRF-like 후처리 수행. 입력 이미지와 모델의 소프트맥스 확률 맵을 사용해 픽셀 단위 클래스 확률을 계산.
        
        Args:
        image (np.ndarray): 원본 이미지 (H, W, 3)
        probmap (np.ndarray): 모델이 출력한 소프트맥스 확률 맵 (C, H, W) - 클래스별 확률

        Returns:
        Q (np.ndarray): CRF-like 후처리된 확률 맵 (C, H, W)
        """
        C, H, W = probmap.shape
        refined_probmap = np.zeros_like(probmap)

        # 각 클래스에 대해 CRF-like 후처리 적용
        for c in range(C):
            class_map = probmap[c]
            
            # 가우시안 필터 적용
            gaussian_filtered = cv2.GaussianBlur(class_map, (0, 0), self.pos_xy_std) * self.pos_w
            
            # 양방향 필터 적용
            bilateral_filtered = cv2.bilateralFilter((class_map * 255).astype(np.uint8), 
                                                     d=0, 
                                                     sigmaColor=self.bi_rgb_std, 
                                                     sigmaSpace=self.bi_xy_std)
            bilateral_filtered = (bilateral_filtered / 255.0) * self.bi_w
            
            # 두 필터 결과를 합산하여 후처리 결과 생성
            refined_probmap[c] = gaussian_filtered + bilateral_filtered

        # 클래스별 확률 맵을 softmax-like 형태로 조정
        Q = np.exp(refined_probmap) / np.sum(np.exp(refined_probmap), axis=0)
        
        return Q