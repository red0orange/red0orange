import os

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
    assert len(image_paths) == len(target_txt_paths) == len(predict_txt_paths)

    image_save_root = os.path.join(save_root, "images")
    target_save_root = os.path.join(save_root, "target")
    predict_save_root = os.path.join(save_root, "predict")

    image_new_paths = []
    target_new_paths = []
    predict_new_paths = []
    # copy 
    for image_path in image_paths:
        shutil.copy(image_path, os.path.join(image_save_root, os.path.basename(image_path)))
        image_new_paths.append(os.path.basename(image_path))
    for target_path in target_txt_paths:
        shutil.copy(target_path, os.path.join(target_save_root, os.path.basename(target_path)))
        target_new_paths.append(os.path.basename(target_path))
    for predict_path in predict_txt_paths:
        shutil.copy(predict_path, os.path.join(predict_save_root, os.path.basename(predict_path)))
        predict_new_paths.append(predict_path)

    csv_data = zip(image_new_paths, target_new_paths, predict_new_paths)
    write_csv(csv_data, os.path.join(save_root, "evaluate_data.csv"))
    pass


class EvaluateData(object):
    """封装了用于评估的数据对象
    """
    def __init__(self, image_paths, target_txt_paths, predict_txt_paths) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.target_txt_paths = target_txt_paths
        self.predict_txt_paths = predict_txt_paths

        pass

    @classmethod
    def create_data_from_csv(cls, csv_path):
        """输入csv生成对象

        Args:
            csv_path (str): 

        Returns:
            EvaluateData: 
        """
        csv_data = read_csv(csv_path)
        image_paths, target_txt_paths, predict_txt_paths = [i[0] for i in csv_data], [i[1] for i in csv_data], [i[2] for i in csv_data]
        return cls(image_paths, target_txt_paths, predict_txt_paths)
    
    @classmethod
    def create_data_from_root(cls, image_root, target_txt_root, predict_txt_root):
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
        assert len(image_paths) == len(target_txt_paths) == len(predict_txt_paths)
        return cls(image_paths, target_txt_paths, predict_txt_paths)