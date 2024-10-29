import torch
import densecrf_gpu

class DenseCRFLayer(torch.nn.Module):
    def __init__(self, W, H, num_classes):
        super(DenseCRFLayer, self).__init__()
        self.W = W
        self.H = H
        self.num_classes = num_classes

    def forward(self, unary, image, num_iterations=10):
        # GPU에서 DenseCRF inference 실행
        return densecrf_gpu.densecrf_inference(unary, image, self.num_classes, num_iterations)