import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val += val * n
        self.count += n
        self.sum = self.val / self.count

def intersectionAndUnionGPU(output, target, num_classes):
    output = output.view(-1)
    target = target.view(-1)
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=num_classes)
    area_output = torch.histc(output, bins=num_classes)
    area_target = torch.histc(target, bins=num_classes)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union


def calculate_mIoU(pred, target, num_classes):
    smooth = 1e-10  # 나누기 0을 방지하기 위한 작은 값
    ious = []
    
    pred = pred.view(-1)  # 예측 값을 1D 텐서로 변환
    target = target.view(-1)  # 실제 값을 1D 텐서로 변환

    for cls in range(num_classes):  # 각 클래스에 대해 IoU 계산
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds[target_inds]).sum().float()
        union = pred_inds.sum().float() + target_inds.sum().float() - intersection
        
        if union == 0:
            ious.append(float('nan'))  # 클래스가 없는 경우는 제외
        else:
            ious.append((intersection + smooth) / (union + smooth))

    return torch.tensor(ious)  # 각 클래스별 IoU 반환
