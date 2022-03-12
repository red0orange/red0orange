import os
import shutil
from matplotlib.pyplot import axis

import numpy as np

import mmcv

from .metrics import *
from ..file import *


class EvaluateHelper(object):
    def __init__(self, image_paths, scores, labels) -> None:
        self.image_path = image_paths
        self.pred_scores = scores
        self.target_labels = labels
        pass

    def cal_metric(self, combine_class_csv=None):
        scores = np.array(self.pred_scores)
        target_labels = np.array(self.target_labels)
        metric_dict = cal_fix_metric(scores.shape[-1], scores, target_labels)

        # 计算合并类别指标
        pred_scores = np.max(scores, axis=1)
        pred_labels = np.argmax(scores, axis=1)
        if combine_class_csv is not None:
            combine_class_data = read_csv(combine_class_csv)
            combine_class_dict = {int(key): int(value) for key, value in combine_class_data}
            combine_class_ids = range(len(set(combine_class_dict.values())))
            combine_gt_labels = np.array([combine_class_dict[key] for key in target_labels])
            combine_pred_labels = np.array([combine_class_dict[key] for key in pred_labels])
            combine_scores = []
            for combine_class_id in range(len(set(combine_class_dict.values()))):
                related_class_ids = [i for i in combine_class_dict.keys() if combine_class_dict[i] == combine_class_id]
                combine_scores.append(np.max(scores[:, related_class_ids], axis=1))
                # combine_pred_scores.append(np.sum(preds[:, related_class_ids], axis=1))
            combine_scores = np.hstack([i[..., None] for i in combine_scores])
            combine_metric_dict = cal_fix_metric(len(combine_class_ids), combine_scores, combine_gt_labels)

            for key, value in combine_metric_dict.items():
                metric_dict["combine " + key] = value
        return metric_dict

    @classmethod
    def create_data_from_mmcv(cls, data_path, *args, **kwargs):
        data = mmcv.load(data_path)
        image_paths = data["image_paths"]
        scores = data["scores"]
        gt_labels = data["target_labels"]
        return cls(image_paths, scores, gt_labels, *args, **kwargs)