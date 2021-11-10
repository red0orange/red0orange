import os
from PIL import Image

from .utils import *
from .file_helper import parse_txt_to_array

from ..file import *


def pack_valid_data_to_evaluate(valid_csv_path, predict_txt_root, save_root):
    """处理我自己的csv格式的目标检测数据集，生成可用于evluate的格式

    Args:
        valid_csv_path (str): 
        predict_txt_root (str): 
        save_root (str): 
    """
    import shutil

    valid_data = read_csv(valid_csv_path)
    image_paths, target_txt_paths = [i[0] for i in valid_data], [i[1] for i in valid_data]
    predict_txt_paths = get_files(predict_txt_root, extensions=[".txt"])

    # 确定数据没问题
    assert len(image_paths) == len(target_txt_paths) == len(predict_txt_paths), "{} != {} != {}".format(len(image_paths), len(target_txt_paths), len(predict_txt_paths))
    image_names = [os.path.basename(i).rsplit(".", maxsplit=1)[0] for i in image_paths]
    predict_names = [os.path.basename(i).rsplit(".", maxsplit=1)[0] for i in predict_txt_paths]
    assert set(image_names) == set(predict_names)
    # 为predict的数据排好序
    predict_txt_paths = [os.path.join(predict_txt_root, i + ".txt") for i in image_names]


    image_save_root = os.path.join(save_root, "images")
    os.makedirs(image_save_root, exist_ok=False)
    target_save_root = os.path.join(save_root, "target")
    os.makedirs(target_save_root, exist_ok=False)
    predict_save_root = os.path.join(save_root, "predict")
    os.makedirs(predict_save_root, exist_ok=False)

    image_new_paths = []
    target_new_paths = []
    predict_new_paths = []
    # copy 
    for image_path in image_paths:
        shutil.copy(image_path, os.path.join(image_save_root, os.path.basename(image_path)))
        image_new_paths.append(os.path.join(image_save_root, os.path.basename(image_path)))
    for target_path in target_txt_paths:
        shutil.copy(target_path, os.path.join(target_save_root, os.path.basename(target_path)))
        target_new_paths.append(os.path.join(target_save_root, os.path.basename(target_path)))
    for predict_path in predict_txt_paths:
        shutil.copy(predict_path, os.path.join(predict_save_root, os.path.basename(predict_path)))
        predict_new_paths.append(os.path.join(predict_save_root, os.path.basename(predict_path)))

    csv_data = zip(image_new_paths, target_new_paths, predict_new_paths)
    write_csv(csv_data, os.path.join(save_root, "evaluate_data.csv"))
    pass


class EvaluateData(object):
    """封装了用于评估的数据对象
    """
    # flags
    TARGET = 0
    PREDICT = 1

    XYXY2XYWH = 0
    XYWH2XYXY = 1

    SCALE2REAL = 0
    SCALE2NORMAL = 1

    def __init__(self, image_paths, target_txt_paths, predict_txt_paths, target_boxes_column_order, predict_boxes_column_order) -> None:
        """构造函数

        Args:
            image_paths (list): 图像路径列表
            target_txt_paths (list): 标注txt文件路径列表
            predict_txt_paths (list): 预测结果txt文件路径列表
            target_boxes_column_order (list): 改变标注txt数据的列顺序，保证数据为5列，且顺序为(class_id, crood_1, crood_2, crood_3, crood_4), crood表示仍不确定是xyxy或xywh，例子为[0, 1, 2, 3, 4]代表不需要更换数据顺序
            predict_boxes_column_order (list): 改变预测结果txt数据的列顺序，保证数据为6列，且顺序为(class_id, crood_1, crood_2, crood_3, crood_4, conf), crood表示仍不确定是xyxy或xywh，例子为[0, 1, 2, 3, 4, 5]代表不需要更换数据顺序

        """
        super().__init__()
        self.image_paths = image_paths
        self.target_txt_paths = target_txt_paths
        self.predict_txt_paths = predict_txt_paths

        self.target_boxes = [parse_txt_to_array(i) for i in self.target_txt_paths]
        self.target_boxes = [i[:, target_boxes_column_order] for i in self.target_boxes if len(i) > 0]

        self.predict_boxes = [parse_txt_to_array(i) for i in self.predict_txt_paths]
        self.predict_boxes = [i[:, predict_boxes_column_order] for i in self.predict_boxes if len(i) > 0]
        pass

    @staticmethod
    def apply_map_to_boxes(boxes, func):
        """封装了对boxes的统一操作，主要是对于空boxes的跳过

        Args:
            boxes (list): 
            func (func): 

        Returns:
            list: 
        """
        result_boxes = []
        for box in boxes:
            if len(box) == 0:
                result_boxes.append(box)
                continue
            box = func(box)
            result_boxes.append(box)
        return result_boxes
    
    def apply_crood_transform_to_boxes(self, which_boxes, which_method):
        """对target或是predict的坐标做变换

        Args:
            which_boxes (int): 选择EvaluateData.TARGET或是EvaluateData.PREDICT
            which_method (int): 选择EvaluateData.TARGET或是EvaluateData.PREDICT
        """
        method_func = None
        if which_method == self.XYXY2XYWH:
            method_func = xyxy2xywh
        elif which_method == self.XYWH2XYXY:
            method_func = xywh2xyxy
            
        input_boxes = None
        if which_boxes == self.TARGET:
            input_boxes = self.target_boxes
        elif which_boxes == self.PREDICT:
            input_boxes = self.predict_boxes

        def tmp_func(box):
            box[:, 1:5] = method_func(box[:, 1:5])
            return box
        output_boxes = EvaluateData.apply_map_to_boxes(input_boxes, tmp_func)

        if which_boxes == self.TARGET:
            self.target_boxes = output_boxes
        elif which_boxes == self.PREDICT:
            self.predict_boxes = output_boxes
        pass

    def apply_scale_to_boxes(self, which_boxes, which_method):
        """将boxes的坐标尺度在真实和归一化之间转换

        Args:
            which_boxes ([type]): [description]
        """
        input_boxes = None
        if which_boxes == self.TARGET:
            input_boxes = self.target_boxes
        elif which_boxes == self.PREDICT:
            input_boxes = self.predict_boxes

        method_func = None
        if which_method == self.SCALE2REAL:
            def tmp_func(box, image_width, image_height):
                box[:, [1, 3]] *= image_width
                box[:, [2, 4]] *= image_height
                box[:, 1:5] = box[:, 1:5].astype(np.int)
                return box
            method_func = tmp_func
        elif which_method == self.SCALE2NORMAL:
            def tmp_func(box, image_width, image_height):
                box[:, 1:5] = box[:, 1:5].astype(np.float)
                box[:, [1, 3]] /= image_width
                box[:, [2, 4]] /= image_height
                return box
            method_func = tmp_func
            
        output_boxes = []
        for i, box in enumerate(input_boxes):
            if len(box) == 0:
                output_boxes.append(box)
                continue
            image_width, image_height = Image.open(self.image_paths[i]).size
            box = method_func(box, image_width, image_height)
            output_boxes.append(box)

        if which_boxes == self.TARGET:
            self.target_boxes = output_boxes
        elif which_boxes == self.PREDICT:
            self.predict_boxes = output_boxes
        pass
        

    @classmethod
    def create_data_from_csv(cls, csv_path, *args, **kwargs):
        """输入csv生成对象

        Args:
            csv_path (str): 

        Returns:
            EvaluateData: 
        """
        csv_data = read_csv(csv_path)
        image_paths, target_txt_paths, predict_txt_paths = [i[0] for i in csv_data], [i[1] for i in csv_data], [i[2] for i in csv_data]
        return cls(image_paths, target_txt_paths, predict_txt_paths, *args, **kwargs)
    
    @classmethod
    def create_data_from_root(cls, image_root, target_txt_root, predict_txt_root, *args, **kwargs):
        """输入数据根目录生成对象

        Args:
            image_root (str): 
            target_txt_root (str): 
            predict_txt_root (str): 

        Returns:
            EvaluateData: 
        """
        image_paths = get_image_files(image_root)
        target_txt_paths = get_files(target_txt_root, extensions=[".txt"])
        predict_txt_paths = get_files(predict_txt_root, extensions=[".txt"])
        # 确定数据无问题
        assert len(image_paths) == len(target_txt_paths) == len(predict_txt_paths)
        image_names = [os.path.basename(i).rsplit(".", maxsplit=1)[0] for i in image_paths]
        target_names = [os.path.basename(i).rsplit(".", maxsplit=1)[0] for i in target_txt_paths]
        predict_names = [os.path.basename(i).rsplit(".", maxsplit=1)[0] for i in predict_txt_paths]
        assert set(image_names) == set(target_names) == set(predict_names)
        # 统一顺序
        image_names = [os.path.basename(i).rsplit(".", maxsplit=1)[0] for i in image_paths]
        target_txt_paths = [os.path.join(target_txt_root, i + ".txt") for i in image_names]
        predict_txt_paths = [os.path.join(predict_txt_root, i + ".txt") for i in image_names]

        return cls(image_paths, target_txt_paths, predict_txt_paths, *args, **kwargs )