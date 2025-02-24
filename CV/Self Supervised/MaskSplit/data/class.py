from collections import defaultdict

# Pascal VOC 클래스 ID → 클래스명 매핑
classId2className = {
    1: 'airplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'cat',
    8: 'car',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorcycle',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tv'
}

# Pascal VOC 클래스명 → 클래스 ID 매핑
className2classId = {v: k for k, v in classId2className.items()}

def get_pascal_splits():
    """Pascal VOC 5-fold 분할"""
    split_classes = defaultdict(dict)
    class_list = list(range(1, 21))

    val_splits = [
        list(range(1, 6)),
        list(range(6, 11)),
        list(range(11, 16)),
        list(range(16, 21))
    ]
    
    split_classes[-1]['val'] = class_list  # 전체 클래스 포함
    for i, val_list in enumerate(val_splits):
        split_classes[i]['val'] = val_list
        split_classes[i]['train'] = list(set(class_list) - set(val_list))

    return split_classes