import imp
import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage


def smart_crop_image(image, x1, y1, x2, y2):
    """智能图像crop

    Args:
        image (numpy): 输入图像
        x1 (int): 左上角x坐标
        y1 (int): 左上角y坐标
        x2 (int): 右下角x坐标
        y2 (int): 右下角y坐标

    Returns:
        numpy: crop得到的图像
    """
    assert len(image.shape) == 2 or len(image.shape) == 3
    assert isinstance(x1, int) and isinstance(y1, int) and isinstance(x2, int) and isinstance(y2, int)
    assert x2 > x1 and y2 > y1

    result_image = np.zeros([y2 - y1, x2 - x1], dtype=np.uint8) if len(image.shape) == 2 else np.zeros([y2 - y1, x2 - x1, image.shape[-1]], dtype=np.uint8)
    x_offset = -x1
    y_offset = -y1

    image_width, image_height = image.shape[:2][::-1]
    x1, x2 = map(lambda i: min(max(i, 0), image_width), [x1, x2])
    y1, y2 = map(lambda i: min(max(i, 0), image_height), [y1, y2])

    result_image[y1+y_offset:y2+y_offset, x1+x_offset:x2+x_offset] = image[y1:y2, x1:x2]
    return result_image


def QImage2numpy(image):
    """QImage转numpy图像

    Args:
        image (QImage): 

    Returns:
        numpy: 
    """
    size = image.size()
    s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)  # format 0xffRRGGBB
    arr = np.frombuffer(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))
    return arr