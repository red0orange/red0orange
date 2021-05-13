# encoding: utf-8
"""
@author: red0orange
@file: file.py
@desc: 文件相关的一些通用utils
"""
import os
import csv
import pandas as pd
import mimetypes
from pathlib import Path


def ifnone(a,b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def poxis2str(l):
    return [str(i) for i in l]

def _path_to_same_str(p_fn):
    "path -> str, but same on nt+posix, for alpha-sort only"
    s_fn = str(p_fn)
    s_fn = s_fn.replace('\\','.')
    s_fn = s_fn.replace('/','.')
    return s_fn


def _get_files(parent, p, f, extensions):
    p = Path(p)#.relative_to(parent)
    if isinstance(extensions,str): extensions = [extensions]
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    res = [p/o for o in f if not o.startswith('.')
           and (extensions is None or f'.{o.split(".")[-1].lower()}' in low_extensions)]
    return res


def get_files(path, extensions=None, recurse:bool=False, exclude=None,
              include=None, presort:bool=False, followlinks:bool=False):
    "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)):
            # skip hidden dirs
            if include is not None and i==0:   d[:] = [o for o in d if o in include]
            elif exclude is not None and i==0: d[:] = [o for o in d if o not in exclude]
            else:                              d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(path, p, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return poxis2str(res)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, path, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return poxis2str(res)


def get_image_files(path, extensions=None, recurse:bool=False, exclude=None,include=None, presort:bool=False, followlinks:bool=False):
    image_extensions = set(k for k, v in mimetypes.types_map.items() if v.startswith('image/'))
    extensions = ifnone(extensions, image_extensions)
    return get_files(path, extensions, recurse, exclude, include, presort, followlinks)


def get_image_extensions():
    return list(set(k for k, v in mimetypes.types_map.items() if v.startswith('image/')))


def write_csv(data, csv_path):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        for line in data:
            writer.writerow(line)
    pass


def read_csv(csv_path):
    result = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            result.append(line)
    return result


class DatasetInfo(object):
    # 一个封装类别信息的类，传入类别信息的csv路径，即可有一些方便的函数，自行查看使用
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)
        pass

    def class_transform(self):
        class_transform = self.df[['origin_class_id', 'train_class_id']].to_numpy()
        class_transform = {k: v for k, v in class_transform}
        return class_transform

    def origin_class_dict(self):
        class_transform = self.df[['origin_class_id', 'origin_class_name']].to_numpy()
        class_transform = {k: v for k, v in class_transform}
        return class_transform

    def train_class_dict(self):
        class_transform = self.df[['train_class_id', 'train_class_name']].to_numpy()
        class_transform = {k: v for k, v in class_transform}
        return class_transform

    @staticmethod
    def create_dataset_info_csv_by_class_count(path, ori_class_id, ori_class_name, class_count, class_num_thres=100):
        assert len(ori_class_id) == len(ori_class_name) == len(class_count), '长度不相等不正常'
        train_class_id = []
        i = 1
        for class_id, class_num in zip(ori_class_id, class_count):
            if class_num <= class_num_thres:
                train_class_id.append(0)
            else:
                train_class_id.append(i)
                i += 1
        DatasetInfo.create_dataset_info_csv(path, ori_class_id, ori_class_name, train_class_id, class_count)
        pass

    @staticmethod
    def create_dataset_info_csv(path, ori_class_id, ori_class_name, train_class_id, class_count):
        assert len(ori_class_id) == len(ori_class_name) == len(train_class_id) == len(class_count), '长度不相等不正常'
        df = pd.DataFrame()
        df['origin_class_id'] = ori_class_id
        df['train_class_id'] = train_class_id
        df['ori_class_name'] = ori_class_name
        df['class_count'] = class_count
        df.to_csv(path, index=False)
        pass