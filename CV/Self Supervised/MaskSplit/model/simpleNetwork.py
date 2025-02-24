import torch
import torch.nn as nn
from torch.nn import functional as F
from model.resnet import resnet50, resnet101
import pytorch_lightning as pl
from util import intersecionUnionGpu, batch_intersecionUnionGpu
from vis import make_episode_visualization

def masked_global_pooling(mask, feature_map):
    # mask size = [N-Way, K-Shot, 1, 56, 56]
    mask = mask.float()
    mask = F.interpolate(mask, size=(feature_map.shape[-2], feature_map.shape[-1]))
    expanded_mask = mask.expand_as(feature_map)
    masked = feature_map * expanded_mask  # mask 0 : 배경, mask 1 : 객체
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
            if m.kernel_size[0] > 1:
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

            self.projection1 = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, padding=1),
                nn.GroupNorm(4, 128),
                nn.ReLU(inplace=True)
            )
            self.projection2 = nn.Sequential(
                nn.Conv2d(1024, 128, kernel_size=3, padding=1),
                nn.GroupNorm(4, 128),
                nn.ReLU(inplace=True)
            )
            self.projection3 = nn.Sequential(
                nn.Conv2d(2048, 128, kernel_size=3, padding=1),
                nn.GroupNorm(4, 128),
                nn.ReLU(inplace=True)
            )
            self.dense = nn.Sequential(
                nn.Conv2d(768, 128, kernel_size=3, padding=1),
                nn.GroupNorm(4, 128),
                nn.ReLU(inplace=True)
            )

        if not self.args.use_all_classes:
            self.val_class_IoU = [ClassIoU(self.args.num_classes_val)]
        else:
            self.val_class_IoU = [ClassIoU(self.args.num_classes_val) for _ in range(4)]

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1, bias=False)
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, x):
        support, smask, query = x

        support = support.squeeze(1)
        with torch.no_grad():
            support_f0 = self.layer0(support)
            support_f1 = self.layer1(support_f0)
            support_f2 = self.layer2(support_f1)
            support_f3 = self.layer3(support_f2)
            support_f4 = self.layer4(support_f3)

        query_f0 = self.layer0(query)
        query_f1 = self.layer1(query_f0)
        query_f2 = self.layer2(query_f1)
        query_f3 = self.layer3(query_f2)
        query_f4 = self.layer4(query_f3)

        support_proj2 = self.projection1(support_f2)
        query_proj2 = self.projection1(query_f2)
        support_proj3 = self.projection2(support_f3)
        query_proj3 = self.projection2(query_f3)
        support_proj4 = self.projection3(support_f4)
        query_proj4 = self.projection3(query_f4)

        smask[smask == 255] = 0  # 배경 0
        pooled_support2 = masked_global_pooling(smask, support_proj2)
        pooled_support3 = masked_global_pooling(smask, support_proj3)
        pooled_support4 = masked_global_pooling(smask, support_proj4)

        fused_f2 = torch.cat([query_proj2, pooled_support2], dim=1)
        fused_f3 = torch.cat([query_proj3, pooled_support3], dim=1)
        fused_f4 = torch.cat([query_proj4, pooled_support4], dim=1)

        fused_feature = torch.cat([fused_f2, fused_f3, fused_f4], dim=1)
        final_feature = self.dense(fused_feature)
        output = self.decoder(final_feature)

        return output


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
        self.intersection.zero_()
        self.union.zero_()

    # Lightining Module
    def training_step(self, batch, batch_idx):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch

        y_hat = self((spprt_imgs, spprt_labels, qry_img))
        target = target.long()
        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        area_inter, area_union, area_target = intersecionUnionGpu(preds, target, 2)
        miou = area_inter.sum().float() / area_union.sum().float()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_miou', miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch

        y_hat = self((spprt_imgs, spprt_labels, qry_img))
        target = target.long()
        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        area_inter, area_union, area_target = intersecionUnionGpu(preds, target, 2)
        miou = area_inter.sum().float() / area_union.sum().float()

        y_hat = y_hat.unsqueeze(1)
        target = target.unsqueeze(1)
        inter, union, _ = batch_intersecionUnionGpu(y_hat, target, 2)
        self.val_class_IoU[dataset_idx].update(inter, union, subcls_list)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_miou', miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        if len(self.val_class_IoU) == 1:  # 단일 IoU인 경우
            val_miou = self.val_class_IoU[0].compute()
            self.log('val_miou', val_miou, prog_bar=True, logger=True)
            self.val_class_IoU[0].reset()
        else:  # 여러 개의 IoU인 경우
            for i, calculator in enumerate(self.val_class_IoU):
                val_miou = calculator.compute()
                calculator.reset()
                self.log("val_miou_" + str(i), val_miou, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx, dataset_idx=0):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch

        target = target.long()
        y_hat = self((spprt_imgs, spprt_labels, qry_img))
        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)

        area_inter, area_union, area_target = intersecionUnionGpu(preds, target, 2)
        miou = area_inter.sum().float() / area_union.sum().float()

        y_hat = y_hat.unsqueeze(1)
        target = target.unsqueeze(1)
        inter, union, _ = batch_intersecionUnionGpu(y_hat, target, 2)

        self.val_class_IoU[dataset_idx].update(inter, union, subcls_list)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_miou', miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 시각화 옵션이 활성화된 경우
        if self.args.visualize:
            for i in range(len(qry_img)):
                path = f"{image_paths[0][i].split('.')[0].split('/')[-1]}_{dataset_idx}_{batch_idx}_{i}"
                make_episode_visualization(
                    spprt_imgs[i].cpu().numpy(),
                    qry_img[i].cpu().numpy(),
                    spprt_labels[i].cpu().numpy(),
                    target[i, 0].cpu().numpy(),
                    y_hat[i].cpu().numpy(),
                    path
                )

    
    def on_test_epoch_end(self):
        test_miou = self.val_class_IoU[0].compute()
        self.log('test_miou', test_miou, prog_bar=True, logger=True)
        self.val_class_IoU[0].reset()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return [optimizer]