import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def plt_show_array(image_array, boxes=None, figsize=(7, 7)):
    """用plt显示numpy array图像，图像为BGR空间，且支持其他基础目标检测的绘制任务

    Args:
        image_array (numpy array): 
        boxes (numpy array, optional): 二维数组，列数为4，要求输入格式为xyxy. Defaults to None.
        figsize (tuple, optional): fig size. Defaults to (7, 7).
    """
    fig, axe = plt.subplots(1, 1, figsize=figsize)
    result_array = image_array.copy()

    if boxes is not None:
        boxes = boxes.astype(np.int)
        for box in boxes:
            cv2.rectangle(result_array, (box[0], box[1]), (box[2], box[3]), (0, 127, 255), 2)

    result_array = result_array[..., ::-1]
    axe.imshow(result_array)

    axe.set_xticks([])
    axe.set_yticks([])

    return fig, axe
