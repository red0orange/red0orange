# encoding: utf-8
"""
@author: red0orange
@file: file.py
@desc: 绘图相关的一些通用utils
"""
import torch
import os
from collections import Counter
import numpy as np
import pandas as pd
from numbers import Number

import matplotlib
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
import seaborn as sns
from .file import read_csv


def create_fig(figsize=(8,8), row=1, col=1):
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    axes = fig.subplots(row, col)
    return fig, axes


def plot_bars(axe, x_labels, ys, y_labels=None, bar_width=0.28):
    if not isinstance(ys[0], list):
        ys = [ys]
    if y_labels is None:
        y_labels = [str(i) for i in range(len(ys))]
    if not isinstance(y_labels, list):
        y_labels = [y_labels]
    x_labels = [str(i) for i in x_labels]
    
    bar_num = len(ys)

    x = np.arange(len(x_labels))  # the label locations
    width = bar_num * bar_width  # the width of the bars

    rects = []
    for i in range(bar_num):
        rects.append(axe.bar(x - width/2 + bar_width*(i+0.5), ys[i], width/bar_num, label=y_labels[i]))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axe.set_ylabel('Scores')
    axe.set_xticks(x)
    axe.set_xticklabels(x_labels)
    axe.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            axe.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for rect in rects:
        autolabel(rect)


def plot_confusion_matrix(num_classes, labels, preds, targets, select_classes=None, figsize=(10, 10), axe=None):
    matrix = np.zeros((num_classes, num_classes))
    for pred_cls_id, target_cls_id in zip(preds, targets):
        matrix[pred_cls_id, target_cls_id] += 1
    if select_classes is not None:
        matrix = matrix[select_classes, :][:, select_classes]
        num_classes = len(select_classes)
        labels = np.array(labels)[select_classes]
        
    if axe is None:
        fig, axe = plt.subplots(1, 1, figsize=figsize)
    else:
        fig=None
    
    data = axe.imshow(matrix, cmap=plt.cm.Blues)
    axe.set_xticks(range(num_classes))
    axe.set_xticklabels(labels, rotation=90)
    axe.set_yticks(range(num_classes))
    axe.set_yticklabels(labels)

    # fig.colorbar(data, ax=axe)
    axe.set_xlabel('True Labels')
    axe.set_ylabel('Predicted Labels')
    axe.set_title('Confusion matrix')

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            info = int(matrix[y, x])
            axe.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")
    return fig


def accuracy_numpy(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def get_per_class_metrics(num_classes, labels, preds, targets, print_result=False):
    class_ids = list(range(num_classes))
    class_names = labels
    gts_num = []
    preds_num = []
    precisions = []
    recalls = []
    specificitys = []
    
    # accs = accuracy_numpy(preds, targets, topk=(1, 2, 3))
    # accs = accuracy_numpy(preds, targets, topk=(1, 2, 3))
    
    for class_id in class_ids:
        pred_cls_num = sum(preds == np.array(class_id))
        gt_cls_num = sum(targets == np.array(class_id))
        preds_num.append(pred_cls_num)
        gts_num.append(gt_cls_num)

        # 新的计算方法
        tp_num = sum((preds == targets) & (preds == class_id))
        tn_num = sum((preds != class_id) & (targets != class_id))
        fp_num = sum((preds == class_id) & (preds != targets))
        fn_num = sum((targets == class_id) & (preds != targets))
        precisions.append(tp_num / (tp_num + fp_num))
        recalls.append(tp_num / (tp_num + fn_num))
        specificitys.append(tn_num / (fp_num + tn_num))

        # 旧的计算方法
        # precisions.append(sum(tp == class_id) / pred_cls_num)
        # recalls.append(sum(tp == class_id) / gt_cls_num)
    if print_result:
        print_metrics_result(class_ids, class_names, gts_num, preds_num, precisions, recalls, specificitys, accs)
    return class_ids, class_names, gts_num, preds_num, precisions, recalls, specificitys


def print_metrics_result(class_ids, class_names, gts_num, preds_num, precisions, recalls, specificitys, accs):
    mean_p = float(np.sum([i for i in precisions if not np.isnan(i)]) / len([i for i in gts_num if i != 0]))
    mean_r = float(np.sum([i for i in recalls if not np.isnan(i)]) / len([i for i in gts_num if i != 0]))
    mean_s = float(np.sum([i for i in specificitys if not np.isnan(i)]) / len(gts_num))

    header = ['class', 'gts', 'preds', 'precision', 'recall(sensitivity)', 'specificity', 'acc'] 
    table_data = [header]
    for i in class_ids:
        row_data = [class_names[i], gts_num[i], preds_num[i], f'{precisions[i]:.3f}', f'{recalls[i]:.3f}', f'{specificitys[i]:.3f}', ""]
        table_data.append(row_data)
    table_data.append(['mean', '', '', f'{mean_p:.3f}', f'{mean_r:.3f}', f'{mean_s:.3f}', "\n".join(["{:.4f}" for i in range(len(accs))]).format(*accs)])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print(table.table)
    pass


# 分类数据集类别分布可视化
matplotlib.rc("font",family='Noto Serif CJK JP')
def plot_classification_dataset_class_distrubute(train_csv_path, valid_csv_path, class_txt_path, save_path=None, figsize=(28, 36)):
    train_data = read_csv(train_csv_path)
    valid_data = read_csv(valid_csv_path)
    class_names = np.loadtxt(class_txt_path, dtype=np.object)
    class_dict = {key: class_names[key] for key in range(len(class_names))}
    train_class_distrubute = dict(Counter([int(i[1]) for i in train_data]))
    train_class_distrubute = {key: train_class_distrubute[key] if (train_class_distrubute.get(key) is not None) else 0 for key in class_dict.keys()}
    valid_class_distrubute = dict(Counter([int(i[1]) for i in valid_data]))
    valid_class_distrubute = {key: valid_class_distrubute[key] if (valid_class_distrubute.get(key) is not None) else 0 for key in class_dict.keys()}
    
    train_data_df = pd.DataFrame(
    data=zip(train_class_distrubute.keys(), [class_dict[i] for i in train_class_distrubute.keys()], train_class_distrubute.values(), ["train" for i in range(len(train_class_distrubute))]),
    columns=["class_id", "class_name", "class_cnt", "type"]
    )
    valid_data_df = pd.DataFrame(
        data=zip(valid_class_distrubute.keys(), [class_dict[i] for i in valid_class_distrubute.keys()], valid_class_distrubute.values(), ["valid" for i in range(len(valid_class_distrubute))]),
        columns=["class_id", "class_name", "class_cnt", "type"]
    )
    concat_data_df = pd.concat([train_data_df, valid_data_df])

    fig, axe = plt.subplots(figsize=figsize)
    sns.set_style("darkgrid")
    bar = sns.barplot(data=concat_data_df, y="class_name", x="class_cnt", hue="type", orient="h", ax=axe)
    axe.legend(loc="lower right", prop = {'size':30})
    axe.set_xlabel("类别数量", fontsize=36)
    axe.set_ylabel("")
    axe.tick_params(axis='x', labelsize=30)
    axe.tick_params(axis='y', labelsize=30)

    for class_id in range(len(class_dict)):  
        # 计算text的坐标
        train_x_class_cnt = train_class_distrubute[class_id]
        train_y = class_id - 0.11
        valid_x_class_cnt = valid_class_distrubute[class_id]
        valid_y = class_id + 0.31
        # 实际绘制
        bar.text(train_x_class_cnt + 10, train_y, str(train_x_class_cnt), color="black", ha="left", fontsize=18)
        bar.text(valid_x_class_cnt + 10, valid_y, str(valid_x_class_cnt), color="black", ha="left", fontsize=18)
    if save_path is not None:
        fig.savefig(save_path)
    return fig