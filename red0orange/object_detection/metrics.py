import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import box_iou


def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        # 这个类别的predict indices
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def cal_metrics(num_classes, preds, targets, iou_thres):
    """计算目标检测的关键指标

    Args:
        num_classes (int): 类别数量
        preds (list): 所有预测box，每个元素为二维数组，columns的格式为 (class_id, x1, y1, x2, y2, conf)
        targets (list): 所有真实标签，每个元素为二维数组，columns的格式为 (class_id, x1, y1, x2, y2)
        iou_thres (float): iou_thres

    Returns:
        precision: 
        recall: 
        ap: 
        cls_count:
        pred_count:
    """
    preds = [torch.Tensor(i) for i in preds]
    targets = [torch.Tensor(i) for i in targets]
    preds = [i.cpu() for i in preds]
    targets = [i.cpu() for i in targets]

    all_tp = []
    all_pred_conf = []
    all_pred_cls = []
    all_target_cls = []

    cls_tp_num = torch.zeros(num_classes, dtype=torch.int)
    cls_fp_num = torch.zeros(num_classes, dtype=torch.int)
    cls_fn_num = torch.zeros(num_classes, dtype=torch.int)
    cls_label_num = torch.zeros(num_classes, dtype=torch.int)
    cls_pred_num = torch.zeros(num_classes, dtype=torch.int)
    assert len(preds) == len(targets), "输入大小不一致"
    for image_i in range(len(preds)):
        pred = preds[image_i]
        labels = targets[image_i]

        # custom statistics per image
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        tp = torch.zeros((pred.shape[0], 1), dtype=torch.int16)
        pred_conf = torch.Tensor([])
        pred_cls = torch.Tensor([])
        target_cls = torch.Tensor([])
        # Predictions
        if len(pred) == 0 or nl == 0:
            if nl:
                # 如果有标注无预测
                target_cls = labels[:, 0]
                for per_label in tcls:
                    cls_fn_num[int(per_label)] += 1
                    cls_label_num[int(per_label)] += 1
                pass
            if len(pred):
                # 如果有预测无标注
                pred_conf = pred[:, 5]
                pred_cls = pred[:, 0]
                for per_pred_cls in pred[:, 5]:
                    cls_fp_num[int(per_pred_cls)] += 1
                    cls_pred_num[int(per_pred_cls)] += 1
                pass
        else:
            # 如果都有，就正常计算下去
            tp = torch.zeros((pred.shape[0], 1), dtype=torch.int16)
            pred_conf = pred[:, 5]
            pred_cls = pred[:, 0]
            target_cls = labels[:, 0]

            predn = pred.clone()
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = labels[:, 1:5]

            # 直接遍历
            detected_set = set()
            detected_pred = []
            if pred.shape[0]:
                ious, i = box_iou(predn[:, 1:5], tbox).max(1)  # ious是每个pred box与最匹配的target box的iou，i是最匹配target box的index
                for j in (ious > iou_thres).nonzero(as_tuple=False):  # 遍历每个拥有匹配target box的pred box
                    if pred[j, 0] == tcls_tensor[i[j]] and (i[j].item() not in detected_set):  # 如果类别匹配且该target box没被匹配过
                        detected_set.add(i[j].item())
                        detected_pred.append(j)
                        cls_tp_num[int(pred[j, 0].item())] += 1
                        tp[j] = 1
                        if len(detected_set) == nl:  # all targets already located in image
                            break
                    pass
            for i in range(pred.shape[0]):
                cls_pred_num[int(pred[i, 0].item())] += 1
                if i not in detected_pred:
                    cls_fp_num[int(pred[i, 0].item())] += 1
                pass
            for i in range(tcls_tensor.shape[0]):
                cls_label_num[int(tcls_tensor[i].item())] += 1
                if i not in detected_set:
                    cls_fn_num[int(tcls_tensor[i].item())] += 1
                pass
        all_tp.append(tp)
        all_pred_cls.append(pred_cls)
        all_pred_conf.append(pred_conf)
        all_target_cls.append(target_cls)

    # precision = np.array(cls_tp_num / (cls_tp_num + cls_fp_num))
    # recall    = np.array(cls_tp_num / (cls_tp_num + cls_fn_num))
    # cls_count = np.array(cls_label_num)
    # pred_count = np.array(cls_pred_num)

    # 改为用官方的方法
    all_tp = torch.cat(all_tp, dim=0)
    all_pred_cls = torch.cat(all_pred_cls, dim=0)
    all_pred_conf = torch.cat(all_pred_conf, dim=0)
    all_target_cls = torch.cat(all_target_cls, dim=0)

    p, r, ap, f1, ap_class = ap_per_class(all_tp, all_pred_conf, all_pred_cls, all_target_cls, plot=False)
    ap_class = ap_class.tolist()

    return ap_class, p, r, ap


def error_analysis_basic(num_classes, preds, targets, iou_thres):
    preds = [torch.Tensor(i) for i in preds]
    targets = [torch.Tensor(i) for i in targets]
    preds = [i.cpu() for i in preds]
    targets = [i.cpu() for i in targets]

    all_tp = []
    all_pred_conf = []
    all_pred_cls = []
    all_target_cls = []
    all_have_match_pred = []
    all_have_match_target = []
    all_pred_match_target = []
    all_ious = []

    cls_tp_num = torch.zeros(num_classes, dtype=torch.int)
    cls_fp_num = torch.zeros(num_classes, dtype=torch.int)
    cls_fn_num = torch.zeros(num_classes, dtype=torch.int)
    cls_label_num = torch.zeros(num_classes, dtype=torch.int)
    cls_pred_num = torch.zeros(num_classes, dtype=torch.int)
    assert len(preds) == len(targets), "输入大小不一致"
    for image_i in range(len(preds)):
        pred = preds[image_i]
        labels = targets[image_i]

        # custom statistics per image
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        
        have_match_pred = torch.zeros((pred.shape[0]), dtype=torch.int16)
        have_match_target = torch.zeros((labels.shape[0]), dtype=torch.int16)
        pred_match_target = torch.zeros((pred.shape[0]), dtype=torch.int64)
        ious = torch.zeros([])
        tp = torch.zeros((pred.shape[0]), dtype=torch.int16)
        pred_conf = torch.Tensor([])
        pred_cls = torch.Tensor([])
        target_cls = torch.Tensor([])
        
        # Predictions
        if len(pred) == 0 or nl == 0:
            if nl:
                # 如果有标注无预测
                target_cls = labels[:, 0]
                for per_label in tcls:
                    cls_fn_num[int(per_label)] += 1
                    cls_label_num[int(per_label)] += 1
                pass
            if len(pred):
                # 如果有预测无标注
                pred_conf = pred[:, 5]
                pred_cls = pred[:, 0]
                for per_pred_cls in pred[:, 5]:
                    cls_fp_num[int(per_pred_cls)] += 1
                    cls_pred_num[int(per_pred_cls)] += 1
                pass
        else:
            # 如果都有，就正常计算下去
            tp = torch.zeros((pred.shape[0], 1), dtype=torch.int16)
            pred_conf = pred[:, 5]
            pred_cls = pred[:, 0]
            target_cls = labels[:, 0]

            predn = pred.clone()
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = labels[:, 1:5]

            # 直接遍历
            detected_set = set()
            detected_pred = []
            if pred.shape[0]:
                ious, i = box_iou(predn[:, 1:5], tbox).max(1)  # ious是每个pred box与最匹配的target box的iou，i是最匹配target box的index
                
                match_index = (ious > iou_thres).nonzero(as_tuple=False)
                have_match_pred[match_index] = 1
                pred_match_target[match_index] = i[match_index]
                have_match_target[i[match_index]] = 1
                
                for j in (ious > iou_thres).nonzero(as_tuple=False):  # 遍历每个拥有匹配target box的pred box
                    if pred[j, 0] == tcls_tensor[i[j]] and (i[j].item() not in detected_set):  # 如果类别匹配且该target box没被匹配过
                        detected_set.add(i[j].item())
                        detected_pred.append(j)
                        cls_tp_num[int(pred[j, 0].item())] += 1
                        tp[j] = 1
                        if len(detected_set) == nl:  # all targets already located in image
                            break
                    pass
            for i in range(pred.shape[0]):
                cls_pred_num[int(pred[i, 0].item())] += 1
                if i not in detected_pred:
                    cls_fp_num[int(pred[i, 0].item())] += 1
                pass
            for i in range(tcls_tensor.shape[0]):
                cls_label_num[int(tcls_tensor[i].item())] += 1
                if i not in detected_set:
                    cls_fn_num[int(tcls_tensor[i].item())] += 1
                pass
        all_tp.append(tp)
        all_pred_cls.append(pred_cls)
        all_pred_conf.append(pred_conf)
        all_target_cls.append(target_cls)
        all_have_match_pred.append(have_match_pred)
        all_have_match_target.append(have_match_target)
        all_pred_match_target.append(pred_match_target)
        all_ious.append(ious)

    # precision = np.array(cls_tp_num / (cls_tp_num + cls_fp_num))
    # recall    = np.array(cls_tp_num / (cls_tp_num + cls_fn_num))
    # cls_count = np.array(cls_label_num)
    # pred_count = np.array(cls_pred_num)

    return all_tp, all_pred_cls, all_pred_conf, all_target_cls, all_have_match_pred, all_have_match_target, all_pred_match_target, all_ious