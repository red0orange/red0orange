# encoding: utf-8
"""
@author: red0orange
@file: file.py
@desc: 绘图相关的一些通用utils
"""
import torch
import os
import numpy as np

import matplotlib.pyplot as plt
from terminaltables import AsciiTable
import seaborn as sns


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
    
    data = axe.imshow(matrix, cmap=plt.cm.Blues)
    axe.set_xticks(range(num_classes))
    axe.set_xticklabels(labels, rotation=45)
    axe.set_yticks(range(num_classes))
    axe.set_yticklabels(labels)

    fig.colorbar(data, ax=axe)
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
    fig.tight_layout()
    return fig


def get_per_class_metrics(num_classes, labels, preds, targets, print_result=True):
    class_ids = list(range(num_classes))
    class_names = labels
    gts_num = []
    preds_num = []
    precisions = []
    recalls = []
    
    tp = preds[preds == targets]
    acc = len(tp) / len(preds)
    
    for class_id in class_ids:
        pred_cls_num = sum(preds == np.array(class_id))
        gt_cls_num = sum(targets == np.array(class_id))
        preds_num.append(pred_cls_num)
        gts_num.append(gt_cls_num)
        precisions.append(sum(tp == class_id) / pred_cls_num)
        recalls.append(sum(tp == class_id) / gt_cls_num)
    
    if print_result:
        header = ['class', 'gts', 'preds', 'precision', 'recall', 'acc'] 
        table_data = [header]
        for i in class_ids:
            row_data = [class_names[i], gts_num[i], preds_num[i], f'{precisions[i]:.3f}', f'{recalls[i]:.3f}', ""]
            table_data.append(row_data)
        table_data.append(['mean', '', '', f'{np.mean(precisions):.3f}', f'{np.mean(recalls):.3f}', f'{acc:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print(table.table)
    return class_ids, class_names, gts_num, preds_num, precisions, recalls