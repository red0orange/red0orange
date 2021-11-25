# encoding: utf-8
"""
@author: red0orange
@file: file.py
@desc: 其他一些utils
"""
from terminaltables import AsciiTable
import csv


def to_format_str(num):
    return '{:^7.3f}'.format(num)


def get_table_str(table_data):
    return AsciiTable(table_data).table


def read_csv(csv_path):
    "读出csv变成一个二维列表"
    result = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 0:
                result.append(row)
    return result


class BaseMeter(object):
    def __init__(self, name):
        self.name = name
        pass

    def return_data(self):
        raise BaseException('must override this func')


class AverageMeter(BaseMeter):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        super(AverageMeter, self).__init__(name)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def return_data(self):
        return self.avg


class AccMeter(BaseMeter):
    def __init__(self, name):
        super(AccMeter, self).__init__(name=name)
        self.predict_all_count = -1
        self.predict_true_count = -1
        self.acc = -1
        pass

    def reset(self):
        self.predict_all_count = -1
        self.predict_true_count = -1
        self.acc = -1
        pass

    def update(self, true_count, all_count):
        self.predict_all_count += all_count
        self.predict_true_count += true_count
        self.acc = self.predict_true_count / self.predict_all_count
        pass

    def return_data(self):
        return self.acc


# 数学方面
def get_list_intersection(listA, listB):
    return [i for i in listA if i in listB]


def get_list_subtraction(listA, listB):
    """
    求差集，得到listB中有而listA中没有的元素
    Args:
        listA:
        listB:

    Returns:

    """
    return list(set(listB).difference(set(listA)))


def get_list_union(listA, listB):
    return list(set(listA).union(set(listB)))


def if_list_contain_list(listA, listB):
    """
    判断listB是否包含listA
    Args:
        listA:
        listB:

    Returns:

    """
    return set(get_list_union(listA, listB)) == set(listB)
