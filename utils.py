import numpy as np

def intersection_and_union(pred, label, num_classes):
    pred = np.argmax(pred, axis=1)  # 예측된 클래스
    pred = pred.flatten()  # 1차원 배열로 변환
    label = label.flatten()  # 1차원 배열로 변환

    intersection = np.histogram2d(label, pred, bins=num_classes, range=[[0, num_classes], [0, num_classes]])[0]
    area_pred = np.histogram(pred, bins=num_classes, range=(0, num_classes))[0]
    area_label = np.histogram(label, bins=num_classes, range=(0, num_classes))[0]

    union = area_pred + area_label - intersection

    return intersection, union
