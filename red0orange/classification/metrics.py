from numbers import Number
import numpy as np

import mmcv


# From mmcls
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


def get_per_class_metrics(num_classes, preds, targets):
    class_ids = list(range(num_classes))
    gts_num = []
    preds_num = []
    precisions = []
    recalls = []
    f1s = []
    specificitys = []
    
    tp = preds[preds == targets]
    acc = len(tp) / len(preds)
    
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
        precision = tp_num / (tp_num + fp_num)
        recall = tp_num / (tp_num + fn_num)
        specificity = tn_num / (fp_num + tn_num)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2*(precision*recall)/(precision+recall))
        specificitys.append(specificity)

        # 旧的计算方法
        # precisions.append(sum(tp == class_id) / pred_cls_num)
        # recalls.append(sum(tp == class_id) / gt_cls_num)
    return np.array(class_ids), np.array(gts_num), np.array(preds_num), np.array(precisions), np.array(recalls), np.array(f1s), np.array(specificitys)


def cal_fix_metric(class_num, scores, gt_labels):
    metric_dict = {}

    top1_acc, top2_acc, top3_acc = accuracy_numpy(scores, gt_labels, topk=(1, 2, 3))
    metric_dict["top1_acc"] = top1_acc
    metric_dict["top2_acc"] = top2_acc
    metric_dict["top3_acc"] = top3_acc

    pred_label = scores.argmax(axis=1)
    pred_score = scores.max(axis=1)
    class_ids, gts_num, preds_num, precisions, recalls, f1s, specificitys = get_per_class_metrics(class_num, pred_label, gt_labels)
    metric_dict["gts_num"] = gts_num
    metric_dict["preds_num"] = preds_num
    metric_dict["precision"] = precisions
    metric_dict["recall"] = recalls
    metric_dict["f1"] = f1s
    metric_dict["specificitys"] = specificitys

    return metric_dict