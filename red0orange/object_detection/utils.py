import math
import json
import torch
import numpy as np
from PIL import Image


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def create_coco_dataset(image_paths, boxes_list, class_dict, save_path):
    """创建coco格式的数据集

    Args:
        image_paths (list): 
        boxes_list (list): 每个box的数据格式为(class_id, x1, y1, x2, y2)，像素单位数据
        class_dict (dict): id map to class name
        save_path (str): json save path
    """
    root_dict = {
        "info": "coco",
        "license": ['none'],
        
        "images": [],
        "annotations": [],
        "categories": []
    }
    # 先添加类别信息
    for class_id, class_name in class_dict.items():
        category_dict = {
            "id": int(class_id),
            "name": class_name
        }
        root_dict["categories"].append(category_dict)
    # 同时添加image和boxes
    assert len(image_paths) == len(boxes_list)
    box_id = 0
    for image_id in range(len(image_paths)):
        image_path = image_paths[image_id]
        boxes = boxes_list[image_id]
        
        image_width, image_height = Image.open(image_path).size
        image_dict = {
            "width": image_width,
            "height": image_height,
            "id": image_id,
            "file_name": image_path  # 这里用完整路径，可以不copy图像
        }
        root_dict["images"].append(image_dict)
        
        for box in boxes: 
            class_id, x1, y1, x2, y2 = box[:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            annotation_dict = {
                "id": box_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": [[1,1, 2,2, 3,3, 4,4, 5,5, 1,1]],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1)
            }
            box_id += 1
            root_dict["annotations"].append(annotation_dict)
    # 导出为json
    with open(save_path, "w") as f:
        json.dump(root_dict, f, indent=1, separators=(',', ': '))
    pass


def create_coco_result_json(boxes_list, save_path):
    """制作COCO类型的result数据

    Args:
        boxes_list (list): 每个box的数据格式为(class_id, x1, y1, x2, y2, score)，像素单位数据
        save_path (str): json保存路径
    """
    root_list = []
    for image_id in range(len(boxes_list)):
        boxes = boxes_list[image_id]
        
        for box in boxes: 
            class_id, x1, y1, x2, y2, conf = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            annotation_dict = {
                "image_id": image_id,
                "category_id": class_id,
                "score": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
            root_list.append(annotation_dict)
    # 导出为json
    with open(save_path, "w") as f:
        json.dump(root_list, f, indent=1, separators=(',', ': '))
    pass