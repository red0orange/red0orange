import os

import matplotlib.pyplot as plt
from PIL import Image

from .utils import *
from .metrics import *
from .file_helper import parse_txt_to_array
from .plot_helper import plt_show_array

from ..plot import plot_confusion_matrix
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

    SCALE2PIXEL = 0
    SCALE2NORMAL = 1

    def __init__(self, image_paths, target_txt_paths, predict_txt_paths, 
                    target_boxes_column_order=(0, 1, 2, 3, 4), pred_boxes_column_order=(0, 1, 2, 3, 4, 5),
                    target_crood_format="xyxy", pred_crood_format="xyxy", 
                    target_scale="pixel", pred_scale="pixel"
                    ) -> None:
        """构造函数

        Args:
            image_paths (list): 图像路径列表
            target_txt_paths (list): 标注txt文件路径列表
            predict_txt_paths (list): 预测结果txt文件路径列表
            target_boxes_column_order (list): 改变标注txt数据的列顺序，保证数据为5列，且顺序为(class_id, crood_1, crood_2, crood_3, crood_4), crood表示仍不确定是xyxy或xywh，例子为[0, 1, 2, 3, 4]代表不需要更换数据顺序
            predict_boxes_column_order (list): 改变预测结果txt数据的列顺序，保证数据为6列，且顺序为(class_id, crood_1, crood_2, crood_3, crood_4, conf), crood表示仍不确定是xyxy或xywh，例子为[0, 1, 2, 3, 4, 5]代表不需要更换数据顺序
            target_crood_format (str): 输入的target box列的顺序，"xyxy"代表格式为(x1, y1, x2, y2)，"xywh"代表格式为(center_x, center_y, width, height)
            pred_crood_format (str): 输入的pred box列的顺序，"xyxy"代表格式为(x1, y1, x2, y2)，"xywh"代表格式为(center_x, center_y, width, height)
            target_scale (str): 输入的target box的尺度，"pixel"代表为像素尺度，"normalize"代表为归一化尺度
            pred_scale (str): 输入的pred box的尺度，"pixel"代表为像素尺度，"normalize"代表为归一化尺度
        """
        super().__init__()
        self.image_paths = image_paths
        self.target_txt_paths = target_txt_paths
        self.predict_txt_paths = predict_txt_paths

        # 转换列的顺序
        self.target_boxes = [parse_txt_to_array(i) for i in self.target_txt_paths]
        self.target_boxes = [i[:, target_boxes_column_order] if len(i) > 0 else i for i in self.target_boxes]
        self.predict_boxes = [parse_txt_to_array(i) for i in self.predict_txt_paths]
        self.predict_boxes = [i[:, pred_boxes_column_order] if len(i) > 0 else i for i in self.predict_boxes]

        # 转换坐标格式
        if target_crood_format == "xyxy": pass
        elif target_crood_format == "xywh":
            self.apply_crood_transform_to_boxes(self.TARGET, self.XYWH2XYXY)
        else: raise BaseException("error input format")
        if pred_crood_format == "xyxy": pass
        elif pred_crood_format == "xywh":
            self.apply_crood_transform_to_boxes(self.PREDICT, self.XYWH2XYXY)
        else: raise BaseException("error input format")

        # 转换尺度
        if target_scale == "pixel": pass
        elif target_scale == "normalized":
            self.apply_scale_to_boxes(self.TARGET, self.SCALE2PIXEL)
        else: raise BaseException("error input format")
        if pred_scale == "pixel": pass
        elif pred_scale == "normalized":
            self.apply_scale_to_boxes(self.PREDICT, self.SCALE2PIXEL)
        else: raise BaseException("error input format")
        pass

    def cal_metrics(self, num_classes, iou_thres, conf_thres):
        return cal_metrics(num_classes, self.predict_boxes, self.target_boxes, iou_thres, conf_thres)

    def cal_fix_metrics(self, num_classes, iou_thres, conf_thres):
        return cal_fix_metric(num_classes, self.predict_boxes, self.target_boxes, iou_thres, conf_thres)

    def show_box(self, image_indexes, pred_indexes=None, target_indexes=None, class_dict=None, show_labels=True, figsize_ratio=1):
        """在选定图中绘制选定矩形框或所有矩形框进行可视化

        Args:
            image_indexes (list): 需要绘制的图像index，每一元素为一个index，如第1、3张图像则为[0, 2]
            pred_indexes (list): 需要绘制的pred的boxes indexes，每一元素为一个boxes indexes，如对应两张图像则为[[1, 2], [2, 3]]，如果元素为-1则全画，如果为空集合则不绘制
            target_indexes (list): 需要绘制的target的boxes indexes，类似pred_indexes，如果元素为-1则全画，如果为空集合则不绘制
            figsize_ratio (int, optional): figsize整体缩放调整的一个比例. Defaults to 1.
        """
        if pred_indexes is None: pred_indexes = [[] for _ in range(len(image_indexes))]
        if target_indexes is None: target_indexes = [[] for _ in range(len(image_indexes))]
        assert len(image_indexes) == len(pred_indexes) == len(target_indexes)
        assert isinstance(image_indexes, list) and isinstance(pred_indexes, list) and isinstance(target_indexes, list)
        assert len(pred_indexes) == 0 or pred_indexes[0] == -1 or isinstance(pred_indexes[0], list)
        assert len(target_indexes) == 0 or target_indexes[0] == -1 or isinstance(target_indexes[0], list)

        image_num = len(image_indexes)
        # max column num is 3
        fig_row = (image_num // 3) + 1
        fig_col = 3 if fig_row > 1 else image_num
        fig, axes = plt.subplots(fig_row, fig_col, figsize=(fig_col * 6 * figsize_ratio, fig_row * 4 * figsize_ratio))
        # fig, axes = plt.subplots(fig_row, fig_col, figsize=(10, 12))
        if isinstance(axes, np.ndarray): axes = axes.flatten().tolist()
        else: axes = [axes]
        for i in range(fig_row * fig_col - image_num):
            fig.delaxes(axes[-(1+i)])

        for i, (image_index, pred_boxes_indexes, target_boxes_indexes) in enumerate(zip(image_indexes, pred_indexes, target_indexes)):
            axe = axes[i]
            image_array = np.array(Image.open(self.image_paths[image_index]))[..., ::-1]
            pred_boxes = self.predict_boxes[image_index]
            target_boxes = self.target_boxes[image_index]
            if len(pred_boxes) == 0: pred_boxes = np.zeros([0, 4])
            if len(target_boxes) == 0: target_boxes = np.zeros([0, 4])

            if pred_boxes_indexes == -1: pred_boxes_indexes = range(len(pred_boxes))
            if target_boxes_indexes == -1: target_boxes_indexes = range(len(target_boxes))

            draw_pred_boxes = pred_boxes[pred_boxes_indexes, 1:5] 
            draw_pred_boxes_ids = pred_boxes[pred_boxes_indexes, 0] 
            draw_pred_boxes_confs = pred_boxes[pred_boxes_indexes, -1] 
            draw_pred_boxes_labels = ["{} {} {} - {:.3f}".format(pred_boxes_indexes[index], int(i), "" if class_dict is None else class_dict[int(i)], j) for index, (i, j) in enumerate(zip(draw_pred_boxes_ids, draw_pred_boxes_confs))]
            draw_target_boxes = target_boxes[target_boxes_indexes, 1:5]
            draw_target_boxes_ids = target_boxes[target_boxes_indexes, 0] 
            draw_target_boxes_labels = ["{} {} {}".format(target_boxes_indexes[index], int(i), "" if class_dict is None else class_dict[int(i)]) for index, i in enumerate(draw_target_boxes_ids)]

            if len(draw_pred_boxes) == 0:  draw_pred_boxes = np.zeros([0, 4])
            if len(draw_target_boxes) == 0: draw_target_boxes = np.zeros([0, 4])
            draw_boxes = np.concatenate([draw_pred_boxes, draw_target_boxes], axis=0)
            draw_color = [(20, 20, 255) for _ in range(len(draw_pred_boxes))] + [(20, 255, 20) for _ in range(len(draw_target_boxes))]
            if show_labels:
                draw_labels = draw_pred_boxes_labels + draw_target_boxes_labels
            else: draw_labels = None
            plt_show_array(image_array, draw_boxes, draw_color, draw_labels, axe=axe)
        
        fig.tight_layout()
        return fig

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
        if which_method == self.SCALE2PIXEL:
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
    def create_data_from_txt(cls, image_root, predict_root, label_root, image_txt, *args, **kwargs):
        image_names = read_txt(image_txt)
        image_paths = [os.path.join(image_root, i) for i in image_names]

        file_names = [i.rsplit(".", maxsplit=1)[0] for i in image_names]
        target_txt_paths = [os.path.join(label_root, i + ".txt") for i in file_names]
        predict_txt_paths = [os.path.join(predict_root, i + ".txt") for i in file_names]
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


class ResultClassAnalyst(object):

    def __init__(self, evaluate_data, class_dict) -> None:
        self.evaluate_data = evaluate_data
        self.class_dict = class_dict

        all_tp, all_pred_cls, all_pred_conf, all_target_cls, all_have_match_pred, all_have_match_target, all_pred_match_target, all_ious = error_analysis_basic(len(self.class_dict), evaluate_data.predict_boxes, evaluate_data.target_boxes, iou_thres=0.6)
        self.all_pred_cls = all_pred_cls
        self.all_pred_conf = all_pred_conf
        self.all_target_cls = all_target_cls
        self.all_have_match_pred = all_have_match_pred
        self.all_have_match_target = all_have_match_target
        self.all_pred_match_target = all_pred_match_target
        self.all_ious = all_ious

        # 构建df来记录所有匹配的box
        i = []
        df_data = []
        df_columns = ["image_name", "pred_cls", "pred_conf", "target_cls", "iou", "pred_cls_id", "target_cls_id", "image_id", "pred_box_id", "target_box_id"]
        for image_i, (pred_cls, pred_conf, target_cls, pred_match_target, ious, have_match_pred) in enumerate(zip(all_pred_cls, all_pred_conf, all_target_cls, all_pred_match_target, all_ious, all_have_match_pred)):
            have_match_indexes = have_match_pred.nonzero(as_tuple=False)
            for index in have_match_indexes:
                i.append("{}_{}".format(image_i, int(pred_match_target[index])))
                df_data.append([
                    os.path.basename(evaluate_data.image_paths[image_i]), 
                    class_dict[int(pred_cls[index])], 
                    float(pred_conf[index]), 
                    class_dict[int(target_cls[pred_match_target[index]])], 
                    float(ious[index]), 
                    int(pred_cls[index]),
                    int(target_cls[pred_match_target[index]]),
                    image_i, 
                    int(index), 
                    int(pred_match_target[index])
                ])
        self.pred_df = pd.DataFrame(data=df_data, index=None, columns=df_columns)

        # 构建target的df
        def analysis_each_target_box(sub_df):
            global debug_i
            if_have_correct_pred_box = False
            correct_pred_box_index = -1
            correct_pred_box_conf = -1
            target_cls = sub_df["target_cls"].iloc[0]
            pred_classes = []
            
            sub_df = sub_df.sort_values(by = 'pred_conf', ascending=True)
            for i, (index, row) in enumerate(sub_df.iterrows()):
                pred_classes.append(row["pred_cls"])
                if row["pred_cls"] == row["target_cls"]:
                    if_have_correct_pred_box = True
                    correct_pred_box_index = len(sub_df) - i
                    correct_pred_box_conf = row["pred_conf"]
            pred_classes = pred_classes[::-1]
            return pd.Series({
                "是否有正确的预测框": if_have_correct_pred_box, 
                "正确预测框的conf排位": correct_pred_box_index, 
                "正确预测框的conf": correct_pred_box_conf, 
                "真实target类别": target_cls,
                "预测框类别": pred_classes
                })
        each_target_box_group = self.pred_df.groupby(["image_id", "target_box_id"])
        self.target_df = each_target_box_group.apply(analysis_each_target_box)
        pass

    def get_no_match_num(self):
        sum_no_match_pred = 0
        sum_pred = 0
        sum_no_match_pred_conf = 0
        for have_match_pred, pred_conf in zip(self.all_have_match_pred, self.all_pred_conf):
            sum_no_match_pred += sum(have_match_pred == 0)
            sum_no_match_pred_conf += sum(pred_conf[have_match_pred == 0])
            sum_pred += len(have_match_pred)
        
        sum_no_match_target = 0
        sum_target = 0
        for have_match_target in self.all_have_match_target:
            sum_no_match_target += sum(have_match_target == 0)
            sum_target += len(have_match_target)
        
        print("无匹配target的pred box数量: {}".format(int(sum_no_match_pred)))
        print("总pred box数量: {}".format(int(sum_pred)))
        print("无匹配pred的target box数量: {}".format(int(sum_no_match_target)))
        print("总target box数量: {}".format(int(sum_target)))
        pass

    def analysis_error_num(self):
        pred_correct_num = sum(self.pred_df["pred_cls_id"] == self.pred_df["target_cls_id"])
        pred_num = len(self.pred_df)
        print("存在匹配target box的所有pred box中，预测类别正确的比例: {:.4f}".format(pred_correct_num / pred_num))

        target_correct_num = sum(self.target_df["正确预测框的conf排位"] == 1)
        target_num = len(self.target_df)
        print("存在匹配pred box的所有target box中，被预测类别正确(置信度第一的pred box预测正确)的比例: {:.4f}".format(target_correct_num / target_num))

        correct_num = sum(self.target_df["是否有正确的预测框"] == True)
        num = len(self.target_df)
        print("存在匹配pred box的所有target box中，存在预测类别正确(不管正确pred box置信度排名)的比例: {:.4f}".format(correct_num / num))

        ratio = sum(self.target_df["正确预测框的conf排位"] == 2) / sum((self.target_df["是否有正确的预测框"] == True) & (self.target_df["正确预测框的conf排位"] != 1))
        mean_score = np.mean(self.target_df[self.target_df["正确预测框的conf排位"] == 2]["正确预测框的conf"])
        print("存在匹配pred box的所有target box中，第二置信度pred box预测正确占所有非第一置信度预测正确pred box的比例: {:.4f}".format(ratio))
        print("存在匹配pred box的所有target box中，第二置信度pred box预测正确的score平均值: {:.4f}".format(mean_score))
        pass

    def plot_target_df_error_confuse_matrix(self, df=None, query=None, select_classes=None, figsize=(15, 15)):
        if df is None: df = self.target_df
        if query is None: query = "正确预测框的conf排位 != 1"
        reverse_class_dict = {key: value for value, key in self.class_dict.items()}
        preds = list(df.query(query)["预测框类别"])
        preds = [reverse_class_dict[i[0]] for i in preds]
        targets = list(df.query(query)["真实target类别"])
        targets = [reverse_class_dict[i] for i in targets]
        fig = plot_confusion_matrix(len(self.class_dict), list(self.class_dict.values()), preds, targets, select_classes=select_classes, figsize=figsize)
        return fig
