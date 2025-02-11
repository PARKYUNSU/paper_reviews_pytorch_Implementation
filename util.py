import torch
import torch.nn as nn
from typing import Tuple


def batch_intersecionUnionGpu(logits, target, num_classes, ignore_index=255):
    n_task, shots, num_classes, h, w = logits.shape
    H, W = target.shape[-2:]

    logits = F.interpolate(logits.view(n_task * shots, num_classes, h, w),
                           size=(H, W), mode='bilinear', align_corners=True).view(n_task, shots, num_classes, H, W)
    preds = logits.argmax(dim=2)

    area_inter = torch.zeros(n_task, shots, num_classes, device=logits.device)
    area_union = torch.zeros(n_task, shots, num_classes, device=logits.device)
    area_target = torch.zeros(n_task, shots, num_classes, device=logits.device)

    for task in range(n_task):
        for shot in range(shots):
            i, u, t = intersecionUnionGpu(preds[task][shot], target[task][shot], num_classes, ignore_index)
            area_inter[task, shot, :] = i
            area_union[task, shot, :] = u
            area_target[task, shot, :] = t
    
    return area_inter, area_union, area_target


def intersecionUnionGpu(preds, target, num_classes, ignore_index=255):
    assert (preds.dim() in [1, 2, 3])
    assert preds.shape == target.shape
    preds = preds.view(-1)
    target = target.view(-1)
    preds[target == ignore_index] = ignore_index
    intersection = preds[preds == target]

    area_inter = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes-1)
    area_output = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes-1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
    area_union = area_output + area_target - area_inter

    return area_inter, area_union, area_target