import torch
import torch.nn as nn
from typing import Tuple
import os
import yaml

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

def load_config(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())