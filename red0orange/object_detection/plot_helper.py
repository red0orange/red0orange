import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def plt_show_array(image_array, boxes=None, color=None, labels=None, axe=None, figsize=(7, 7)):
    """用plt显示numpy array图像，图像为BGR空间，且支持其他基础目标检测的绘制任务

    Args:
        image_array (numpy array): 
        boxes (numpy array, optional): 二维数组，列数为4，要求输入格式为xyxy. Defaults to None.
        figsize (tuple, optional): fig size. Defaults to (7, 7).
    """
    if axe is None:
        fig, axe = plt.subplots(1, 1, figsize=figsize)
    result_array = image_array.copy()

    if boxes is not None:
        if len(boxes.shape) == 1:
            boxes = boxes[None, :]
        boxes_num = len(boxes)

        # deal color
        if color is None:
            color = [[0, 127, 255] for _ in range(boxes_num)]
        elif isinstance(color, list) or isinstance(color, tuple):
            color = list(color)
            if isinstance(color[0], list) or isinstance(color[0], tuple):
                assert len(color) == boxes_num
            else:
                color = [color for _ in range(boxes_num)]
        else:
            assert 0, "color format error."
        
        # deal label
        if labels is None:
            labels = [None for _ in range(boxes_num)]
        else:
            if not isinstance(labels, list):
                labels = [labels for _ in range(boxes_num)]
            else:
                assert len(labels) == boxes_num

        boxes = boxes.astype(np.int)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(result_array, (x1, y1), (x2, y2), color[i], 3)

            if labels[i] is not None:
                label = str(labels[i])
                # For the text background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                # Prints the text.    
                result_array = cv2.rectangle(result_array, (x1, y1 - 20), (x1 + w, y1), color[i], -1)
                result_array = cv2.putText(result_array, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    result_array = result_array[..., ::-1]
    axe.imshow(result_array)

    axe.set_xticks([])
    axe.set_yticks([])

    return axe
