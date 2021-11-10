import torch
import numpy as np

from .utils import box_iou


def cal_metrics(num_classes, preds, targets, iou_thres):
    # input format
    # preds include pred, pred per line (x1, y1, x2, y2, conf, class_id)
    # targets include target, target per line (class_id, x1, y1, x2, y2)

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
        if len(pred) == 0:
            if nl:
                for per_label in tcls:
                    cls_fn_num[int(per_label)] += 1
                    cls_label_num[int(per_label)] += 1
                pass
            continue

        # Predictions
        predn = pred.clone()
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = labels[:, 1:5]

            # 直接遍历
            detected_set = set()
            detected_label = []
            detected_pred = []
            if pred.shape[0]:
                ious, i = box_iou(predn[:, :4], tbox).max(1)  # best ious, indices
                for j in (ious > iou_thres).nonzero(as_tuple=False):
                    if pred[j, 5] == tcls_tensor[i[j]] and (i[j].item() not in detected_set):
                        detected_set.add(i[j])
                        detected_label.append(i[j].item())
                        detected_pred.append(j)
                        cls_tp_num[int(pred[j, 5].item())] += 1
                        if len(detected_label) == nl:  # all targets already located in image
                            break
                    pass
            for i in range(pred.shape[0]):
                cls_pred_num[int(pred[i, 5].item())] += 1
                if i not in detected_pred:
                    cls_fp_num[int(pred[i, 5].item())] += 1
                pass
            for i in range(tcls_tensor.shape[0]):
                cls_label_num[int(tcls_tensor[i].item())] += 1
                if i not in detected_label:
                    cls_fn_num[int(tcls_tensor[i].item())] += 1
                pass

            # 遍历每个类别
            # 从原来计算AP的方法修改得到，注意遍历类别一定要遍历预测、标注的所有类别，如果按照原来的只遍历标注的类别，
            # 在for循环内就只能计算得到TP是正确的，FP和FN要在for循环结束后一次性用总的pred、label进行计算，否则就会
            # 漏掉很多错误的标注。

            # for cls in torch.unique(torch.cat([pred[:, 5], tcls_tensor])):
            #     # 取得这一类别的target indices和predict indices, 因此后面只需要考虑位置匹配
            #     ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
            #     pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
            #     # Search for detections
            #     tp_num = 0
            #     if ti.shape[0]:
            #         if pi.shape[0]:
            #             # Prediction to target ious
            #             # pred_num * 1, 取得每个pred与所有target中最大iou的iou值和对应的target的id
            #             ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
            #
            #             # Append detections
            #             detected_set = set()
            #             for j in (ious > iouv[0]).nonzero(as_tuple=False):
            #                 d = ti[i[j]]  # detected target
            #                 if d.item() not in detected_set:
            #                     detected_set.add(d.item())
            #                     detected.append(d)
            #                     # TODO
            #                     tp_num += 1
            #                     if len(detected) == nl:  # all targets already located in image
            #                         break
            #     cls_tp_num[int(cls.item())] += tp_num
            #     cls_fp_num[int(cls.item())] += (pi.shape[0] - tp_num)
            #     cls_fn_num[int(cls.item())] += (ti.shape[0] - tp_num)
            #     cls_label_num[int(cls.item())] += ti.shape[0]
            #     cls_pred_num[int(cls.item())] += pi.shape[0]

    # 得到最终指标
    precision = np.array(cls_tp_num / (cls_tp_num + cls_fp_num))
    recall    = np.array(cls_tp_num / (cls_tp_num + cls_fn_num))
    cls_count = np.array(cls_label_num)
    pred_count = np.array(cls_pred_num)

    return precision, recall, cls_count, pred_count