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
        
