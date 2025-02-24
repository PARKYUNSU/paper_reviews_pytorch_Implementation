import torch
import torch.nn as nn
import torch.nn.functional as F

from .VGG16_LargeFV import VGG16_LargeFV

import itertools, utils
import numpy as np
from PIL import Image

import pydensecrf.densecrf as dcrf
import pydensecrf.utils
import pydensecrf.utils as crf_utils

class DeepLabv1(nn.Module):
    def __init__(self, num_classes=32, init_weights=True, gpu_id=0, weight_file=None):
        super(DeepLabv1, self).__init__()
        
        self.num_classes = num_classes
        self.gpu = gpu_id
        torch.cuda.set_device(self.gpu)

        # Backbone 모델 초기화
        self.backbone = VGG16_LargeFV(num_classes=num_classes).cuda(self.gpu)
        
        # weight_file이 제공된 경우 가중치 로드
        if weight_file is not None:
            self.backbone.load_state_dict(torch.load(weight_file))
        elif init_weights:
            self._initialize_weights()

        # 입력 정규화를 위한 mean 및 std 정의
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda(self.gpu, non_blocking=True)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda(self.gpu, non_blocking=True)
        
        self.eps = 1e-10
        self.best_mIoU = 0.0

    def _initialize_weights(self):
        # 가중치 초기화 코드
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        probmap = F.softmax(x, dim=1)
        return probmap

    def grid_search(self, data_loader, iter_max, bi_ws, bi_xy_stds, bi_rgb_stds, pos_ws, pos_xy_stds):
        self.eval()
        with torch.no_grad():
            for bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std in itertools.product(bi_ws, bi_xy_stds, bi_rgb_stds, pos_ws, pos_xy_stds):
                
                tps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
                fps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
                fns = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
                
                crf = DenseCRF(iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
                
                for i, (images, labels) in enumerate(data_loader):
                    if i == 100: break
                        
                    images, labels = images.cuda(self.gpu, non_blocking=True), labels.cuda(self.gpu, non_blocking=True)
                    images = images.float().div(255).sub_(self.mean).div_(self.std)
                    
                    outputs = self(images)
                    outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
                    
                    for j in range(images.size(0)):
                        pred = crf(images[j].cpu(), outputs[j].cpu())
                        pred = torch.from_numpy(pred).float().cuda(self.gpu, non_blocking=True)
                        pred = torch.argmax(pred, dim=0)
                        
                        filter_255 = labels[j] != 255
                        for c in range(self.num_classes):
                            tp = (pred == c) & (labels[j] == c) & filter_255
                            fp = (pred == c) & (labels[j] != c) & filter_255
                            fn = (pred != c) & (labels[j] == c) & filter_255
                            tps[c] += tp.sum()
                            fps[c] += fp.sum()
                            fns[c] += fn.sum()
                
                mIoU = (tps / (self.eps + tps + fps + fns)).mean()
                
                state = f'bi_w: {bi_w}, bi_xy_std: {bi_xy_std}, bi_rgb_std: {bi_rgb_std}, pos_w: {pos_w}, pos_xy_std: {pos_xy_std}  mIoU: {100 * mIoU:.4f}'
                
                if mIoU > self.best_mIoU:
                    print("\n" + "*" * 35 + " Best mIoU Updated " + "*" * 35)
                    print(state)
                    self.best_mIoU = mIoU
                else:
                    print(state)
                    
    def inference(self, image_dir, iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std):
        self.eval()
        with torch.no_grad():
            image = Image.open(image_dir).convert('RGB')
            
            image_tensor = torch.as_tensor(np.asarray(image).copy())
            image_tensor = image_tensor.view(image.size[1], image.size[0], len(image.getbands()))
            image_tensor = image_tensor.permute((2, 0, 1))
            
            c, h, w = image_tensor.shape
            image_norm_tensor = image_tensor[None, ...].float().div(255).cuda(self.gpu, non_blocking=True)
            
            image_norm_tensor = image_norm_tensor.sub_(self.mean).div_(self.std)
            
            output = self(image_norm_tensor)
            output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
            output = nn.Softmax2d()(output)
            output = output[0]
            
            crf = DenseCRF(iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
            
            predict = crf(image_tensor, output)
            predict = np.argmax(predict, axis=0)
            return predict
        

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