# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.VGG16_LargeFV import VGG16_LargeFV

# class DeepLabV1(nn.Module):
#     def __init__(self, num_classes=32, init_weights=True):
#         super(DeepLabV1, self).__init__()
#         self.backbone = VGG16_LargeFV(num_classes=num_classes, init_weights=init_weights)

#     def forward(self, x):
#         x = self.backbone(x)
#         probmap = F.softmax(x, dim=1)
        
#         return probmap
import torch
import torch.nn as nn
import torch.nn.functional as F

from .VGG16_LargeFV import VGG16_LargeFV

import itertools, utils
import numpy as np
from PIL import Image

import pydensecrf.densecrf as dcrf
import pydensecrf.utils


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
            
            image_tensor = torch.as_tensor(np.asarray(image))
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
        

class DenseCRF():
    def __init__(self, iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std):
        self.iter_max = iter_max
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std

    def __call__(self, image, prob_map):
        C, H, W = prob_map.shape
        
        image = image.permute((1, 2, 0))
        prob_map = prob_map.cpu().numpy()
        
        U = pydensecrf.utils.unary_from_softmax(prob_map)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)
        
        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w)

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q