import torch
import torch.nn as nn
from torch.nn import functional as F
from model.resnet import resnet50, resnet101
import pytorch_lightning as pl


def masked_global_pooling(mask, feature_map):
    # mask size = [N-Way, K-Shot, 1, 56, 56]
    mask = mask.float()
    mask = F.interpolate(mask, size=(feature_map.shape[-2], feature_map.shape[-1]))
    expanded_mask = mask.expand_as(feature_map)
    masked = feature_map * expanded_mask # mask 0 : 배경, mask 1 : 객체
    out = torch.sum(masked, dim=[-1, -2]) / (expanded_mask.sum(dim=[-1, -2]) + 1e-5)
    # 2D 형태로 변형
    out = out.unsqueeze(-1).unsqueeze(-1)
    out = out.expand_as(feature_map)
    return out


def apply_dilation(layer, dilation_rate):
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            m.dilation = (dilation_rate, dilation_rate)
            m.padding = (dilation_rate, dilation_rate)
            m.stride = (1, 1)


class SimpleNetwork(pl.LightningModule):
    def __init__(self, hparams, visualize=False):
        super(SimpleNetwork, self).__init__()
        print(hparams)
        self.save_hyperparameters()
        self.args = hparams
        self.args.visualize = self.hparams.visualize

        if self.args.arch == 'resnet':
            if self.args.layers == 50:
                resnet = resnet50(pretrained=self.args.pretrained, deep_base=self.args.deep_base)
            else:
                resnet = resnet101(pretrained=self.args.pretrained, deep_base=self.args.deep_base)

            if self.args.deep_base:
                self.layer0 = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu,
                    resnet.conv2, resnet.bn2, resnet.relu,
                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            else:
                self.layer0 = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

            self.layer1, self.layer2, self.layer3, self.layer4 = (
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

            self.feature_res = (50, 50)

            apply_dilation(self.layer3, dilation_rate=2)
            apply_dilation(self.layer4, dilation_rate=4)

            self.project1 = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, padding=1)
                ),nn.GroupNorm(4, 128), nn.ReLU(inplace=True)
            self.project2 = nn.Sequential(
                nn.Conv2d(1024, 128, kernel_size=3, padding=1)
                ),nn.GroupNorm(4, 128), nn.ReLU(inplace=True)
            self.project3 = nn.Sequential(
                nn.Conv2d(2048, 128, kernel_size=3, padding=1)
                ),nn.GroupNorm(4, 128), nn.ReLU(inplace=True)
            self.dense = nn.Sequential(
                nn.Conv2d(768, 128, kernel_size=3, padding=1)
                ),nn.GroupNorm(4, 128), nn.ReLU(inplace=True)
            
            if not self.args.use_all_classes:
                self.val_class_IoU = [ClassIoU(self.args.num_classes_val)]
            else:
                self.val_class_IoU = [ClassIoU(self.args.num_classes_val),
                                    ClassIoU(self.args.num_classes_val),
                                    ClassIoU(self.args.num_classes_val),
                                    ClassIoU(self.args.num_classes_val)]


class ClassIoU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes) # 각 클래스별 교집합
        self.union = torch.zeros(num_classes) # 각 클래스 별 합집합

    def update(self, pred, target):
        for i in range(self.num_classes):
            pred_mask = pred == i
            label_mask = target == i

            intersection = (pred_mask & label_mask).float().sum()
            union = (pred_mask | label_mask).float().sum()

            self.intersection[i] += intersection
            self.union[i] += union

    def get_iou(self):
        return self.intersection / (self.union + 1e-5)
    
    def reset(self):
        self.intersection.zero_(self.num_classes)
        self.union.zero_(self.num_classes)