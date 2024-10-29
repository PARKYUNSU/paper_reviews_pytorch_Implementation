import torch
import torch.nn.functional as F

class DenseCRFLayer:
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std, device='cuda'):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.device = device

    def apply_gaussian_filter(self, class_map):
        # 가우시안 커널 정의
        kernel_size = int(2 * self.pos_xy_std + 1)
        sigma = self.pos_xy_std
        gaussian_kernel = torch.tensor(
            [1.0 / (2.0 * torch.pi * sigma**2) * torch.exp(-((x - kernel_size//2)**2) / (2 * sigma**2))
             for x in range(kernel_size)],
            dtype=torch.float32, device=self.device
        )
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # 정규화
        gaussian_kernel = gaussian_kernel.view(1, 1, -1, -1)  # conv2d에 맞게 reshape

        # 가우시안 필터 적용
        class_map = class_map.unsqueeze(0).unsqueeze(0)  # 배치와 채널 차원 추가
        gaussian_filtered = F.conv2d(class_map, gaussian_kernel, padding=kernel_size//2)
        return gaussian_filtered.squeeze() * self.pos_w

    def apply_bilateral_filter(self, class_map, image):
        # 양방향 필터 근사화
        class_map = class_map.unsqueeze(0).unsqueeze(0)  # 배치와 채널 차원 추가
        image = image.float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0  # 이미지 정규화

        # 양방향 필터 적용 (여기서는 단순히 conv2d로 처리)
        bilateral_filtered = F.conv2d(class_map * 255, kernel_size=0, stride=self.bi_xy_std)  # 임시 적용
        bilateral_filtered = bilateral_filtered / 255.0 * self.bi_w

        return bilateral_filtered.squeeze()

    def forward(self, image, probmap):
        C, H, W = probmap.shape
        refined_probmap = torch.zeros_like(probmap)

        # 각 클래스에 대해 CRF-like 후처리 수행
        for c in range(C):
            class_map = probmap[c]
            
            # 가우시안 필터 적용
            gaussian_filtered = self.apply_gaussian_filter(class_map)

            # 양방향 필터 적용
            bilateral_filtered = self.apply_bilateral_filter(class_map, image)

            # 두 필터 결과를 합산하여 후처리 생성
            refined_probmap[c] = gaussian_filtered + bilateral_filtered

        # 확률 맵 정규화를 위한 softmax 적용
        Q = torch.exp(refined_probmap) / torch.sum(torch.exp(refined_probmap), dim=0)
        return Q
