# encoding: utf-8
"""
@author: red0orange
@file: preprocess.py
@desc: 数据预处理可能会用到的utils
"""
import os
import csv
import numpy as np
import xml.dom.minidom as minidom
from collections import Counter, OrderedDict
import warnings


def modify_xml_to_correct(root_path, new_root, class_dict):
    "临时函数更改所有xml文件用于适应代码的变化"
    paths = get_files(root_path, ['.xml'], recurse=True)
    for path in paths:
        try:
            rel_path = os.path.relpath(path, root_path)
            old_path = path
            new_path = os.path.join(new_root, rel_path)
            print(old_path)
            if not os.path.isdir(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
            with open(old_path, 'r', encoding='utf-8') as f:
                dom = minidom.parse(f)
                root = dom.documentElement

                filename = root.getElementsByTagName('filename')[0].firstChild.data
                fileLocation_ = root.getElementsByTagName('fileLocation')[0].firstChild.data
                relativeLocation = root.getElementsByTagName('relativeLocation')[0].firstChild.data
                img_width_ = root.getElementsByTagName('width')[0].firstChild.data
                img_height_ = root.getElementsByTagName('height')[0].firstChild.data

                object = root.getElementsByTagName('object')
                if len(object) != 1:
                    raise BaseException('xml error')
                object = object[0]
                names = object.getElementsByTagName('name')
                names = [name_node.firstChild.data for name_node in names]
                xs = object.getElementsByTagName('x')
                xs = [x_node.firstChild.data for x_node in xs]
                ys = object.getElementsByTagName('y')
                ys = [y_node.firstChild.data for y_node in ys]
                widths = object.getElementsByTagName('width')
                widths = [width_node.firstChild.data for width_node in widths]
                heights = object.getElementsByTagName('height')
                heights = [hegiht_node.firstChild.data for hegiht_node in heights]

            # 重新写
            doc = minidom.Document()
            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)

            file_path = doc.createElement('file_path')
            file_path.appendChild(doc.createTextNode(fileLocation_))
            annotation.appendChild(file_path)

            OneDirname = doc.createElement('patientNum')  # 上一级的目录名
            OneDirname.appendChild(doc.createTextNode(os.path.dirname(fileLocation_).split('/')[-1]))
            TwoDirname = doc.createElement('illSort')  # 上二级的目录名
            TwoDirname.appendChild(doc.createTextNode(os.path.dirname(fileLocation_).split('/')[-2]))
            annotation.appendChild(OneDirname)
            annotation.appendChild(TwoDirname)

            size = doc.createElement('img_size')
            annotation.appendChild(size)
            width = doc.createElement('width')
            width.appendChild(doc.createTextNode('{}'.format(img_width_)))
            height = doc.createElement('height')
            height.appendChild(doc.createTextNode('{}'.format(img_height_)))
            size.appendChild(width)
            size.appendChild(height)

            for x_, y_, width_, height_, name_ in zip(xs, ys, widths, heights, names):
                _object = doc.createElement('object')
                annotation.appendChild(_object)

                # 已经错误了的名字
                name_ = name_.replace('淋巴细胞', '幼淋巴细胞').replace('嗜酸晚幼粒细胞', '嗜酸性粒细胞')\
                    .replace('幼幼淋巴细胞', '幼淋巴细胞').replace('异型幼淋巴细胞', '幼淋巴细胞').replace('原幼稚幼淋巴细胞', '原幼稚淋巴细胞')\
                    .replace('成熟幼淋巴细胞', '成熟淋巴细胞').replace('晩幼粒细胞', '晚幼粒细胞')

                x_, y_, width_, height_, img_width_, img_height_ = map(int, [x_, y_, width_, height_, img_width_,
                                                                             img_height_])
                x_, width_ = map(lambda i: '{:.3}'.format(i / img_width_), [x_, width_])
                y_, height_ = map(lambda i: '{:.3}'.format(i / img_height_), [y_, height_])
                categories_id = doc.createElement('categories_id')
                categories_id.appendChild(doc.createTextNode(str(class_dict[name_])))
                _object.appendChild(categories_id)
                name = doc.createElement('name')
                name.appendChild(doc.createTextNode(name_))
                _object.appendChild(name)
                x = doc.createElement('x')
                x.appendChild(doc.createTextNode(x_))
                y = doc.createElement('y')
                y.appendChild(doc.createTextNode(y_))
                width = doc.createElement('width')
                width.appendChild(doc.createTextNode(width_))
                height = doc.createElement('height')
                height.appendChild(doc.createTextNode(height_))
                _object.appendChild(x)
                _object.appendChild(y)
                _object.appendChild(width)
                _object.appendChild(height)

            with open(new_path, 'w', encoding='utf-8') as f:
                doc.writexml(f, indent='', addindent='\t', newl='\n', encoding='utf-8')
        except:
            print(old_path)

            
def parse_xml_to_boxes(xml_path):
    boxes = []
    names = []
    with open(xml_path, 'r', encoding='utf-8') as f:
        dom = minidom.parse(f)
        root = dom.documentElement
        objects = root.getElementsByTagName('object')
        for object_ in objects:
            categories_id = object_.getElementsByTagName('categories_id')[0].firstChild.data
            name = object_.getElementsByTagName('name')[0].firstChild.data
            x = object_.getElementsByTagName('x')[0].firstChild.data
            y = object_.getElementsByTagName('y')[0].firstChild.data
            width = object_.getElementsByTagName('width')[0].firstChild.data
            height = object_.getElementsByTagName('height')[0].firstChild.data
            categories_id = int(categories_id)
            x, y, width, height = map(float, [x, y, width, height])
            boxes.append([int(categories_id), x, y, width, height])
            names.append(name)
    return np.array(boxes), names
            

def change_xml(old_root, new_root, xml_paths, class_transform=None):
    "转换新的xml格式的函数"
    for xml_path in xml_paths:
        rel_path = os.path.relpath(xml_path, old_root)
        old_path = xml_path
        new_path = os.path.join(new_root, rel_path)
#         print(new_path)
        new_path = new_path.rsplit('.', maxsplit=1)[0]+'.txt'
        if not os.path.isdir(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
        boxes, names = parse_xml_to_boxes(old_path)

        # 转换格式
        if len(boxes.shape) == 1:
            print('empty label: ', old_path)
            boxes = boxes[None, ...]
            with open(new_path, 'w') as txt:
                pass
            continue
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 2] + boxes[:, 4] / 2
#         print(new_path)
        with open(new_path, 'w') as txt:
            for box_i in range(boxes.shape[0]):
                cat_id, x, y, width, height = boxes[box_i, :]
                if class_transform:
                    cat_id = class_transform[cat_id]
                txt.write('{} {:.3} {:.3} {:.3} {:.3}'.format(cat_id, x, y, width, height) + '\n')


def get_valid_image_label(image_root, lebel_root):
    "从图片的根目录和label的根目录获得无重名的image->label的文件对应字典"
    image_extensions = set(k for k, v in mimetypes.types_map.items() if v.startswith('image/'))
    img_paths = get_files(image_root, extensions=image_extensions, recurse=True)
    label_paths = get_files(lebel_root, extensions=['.txt'], recurse=True)

    img_names = [os.path.basename(path).split('.')[0] for path in img_paths]
    label_names = [os.path.basename(path).split('.')[0] for path in label_paths]

    img_label = {}
    for i, img_name in enumerate(img_names):
        img_label[img_paths[i]] = []
        for ii, label_name in enumerate(label_names):
            if img_name == label_name:
                img_label[img_paths[i]].append(label_paths[ii])

    valid_img_label = {k: v[0] for k, v in img_label.items() if len(v) == 1}
    return valid_img_label


def get_samples_statistics(image_paths, label_paths):
    "传入对应的图片路径列表和标注路径列表"
    image_info_result = {}
    class_info_result = {}
    for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        boxes = np.loadtxt(label_path)
        if len(boxes) == 0:
            image_info_result[image_path] = {}
            continue
        if len(boxes.shape) == 1: boxes = boxes[None, ...]
        labels = boxes[:, 0].astype(np.int64)
        cls_counter = dict(Counter(labels))
        image_info_result[image_path] = cls_counter
        for key, value in cls_counter.items():
            if key not in class_info_result.keys():
                class_info_result[key] = {}
            class_info_result[key][image_path] = value
    return image_info_result, class_info_result


def per_class_split(class_info_result, image_info_result, test_ratio=0.15, valid_ratio=0.1):
    train_image_paths = []
    test_image_paths = []
    valid_image_paths = []

    sorted_class_info = [[cls_id, info] for cls_id, info in class_info_result.items()]
    sorted_class_info = sorted(sorted_class_info, key=lambda k: sum(list(k[1].values())))

    used_image_paths = []

    # 得到每个类别每个数据集需要的数量
    cls_split = [{}, {}, {}, {}]  # all   train  test  valid
    for cls_id, info in sorted_class_info:
        # sort the dict
        cls_num = 0
        sorted_info = []
        for image_path, num in info.items():
            sorted_info.append([image_path, num])
            cls_num += num
        sorted_info = sorted(sorted_info, key=lambda k: k[1])

        train_ratio = 1 - test_ratio - valid_ratio
        train_num = round(cls_num * train_ratio)
        test_num = round((cls_num - train_num) * (test_ratio / (test_ratio + valid_ratio)))
        valid_num = round((cls_num - train_num) * (valid_ratio / (test_ratio + valid_ratio)))

        cls_split[0][cls_id] = cls_num
        cls_split[1][cls_id] = train_num
        cls_split[2][cls_id] = test_num
        cls_split[3][cls_id] = valid_num

    # 不断更新的实际分布情况
    res_cls_split = [cls_split[0],
                     {k: 0 for k in cls_split[1].keys()},
                     {k: 0 for k in cls_split[2].keys()},
                     {k: 0 for k in cls_split[3].keys()}]
    # 投票分每张图片
    for i, (image_path, info) in enumerate(image_info_result.items()):
        # 得到这张图片的类别情况
        exist_cls, exist_cls_num = info.keys(), info.values()
        if len(exist_cls) == 0:
            warnings.warn('empty image: {}'.format(image_path))
            continue
        # 计算谁最合适得到这张图片
        train_error_all, test_error_all, valid_error_all = [], [], []
        for cls_id in exist_cls:
            if cls_split[1][cls_id] != 0:
                train_error_all.append(
                    100 * abs((cls_split[1][cls_id] - res_cls_split[1][cls_id]) / cls_split[1][cls_id]))
            if cls_split[2][cls_id] != 0:
                test_error_all.append(
                    100 * abs((cls_split[2][cls_id] - res_cls_split[2][cls_id]) / cls_split[2][cls_id]))
            if cls_split[3][cls_id] != 0:
                valid_error_all.append(
                    100 * abs((cls_split[3][cls_id] - res_cls_split[3][cls_id]) / cls_split[3][cls_id]))
        train_mean, test_mean, valid_mean = map(lambda k: sum(k)/len(k) if len(k) != 0 else 0,
                                                [train_error_all, test_error_all, valid_error_all])
        if max([train_mean, test_mean, valid_mean]) == train_mean:
            suitable_dataset = train_image_paths
            suitable_cls_split = res_cls_split[1]
        elif max([train_mean, test_mean, valid_mean]) == test_mean:
            suitable_dataset = test_image_paths
            suitable_cls_split = res_cls_split[2]
        elif max([train_mean, test_mean, valid_mean]) == valid_mean:
            suitable_dataset = valid_image_paths
            suitable_cls_split = res_cls_split[3]
        suitable_dataset.append(image_path)
        for cls_id, cls_num in zip(exist_cls, exist_cls_num):
            suitable_cls_split[cls_id] += cls_num
    return train_image_paths, valid_image_paths, test_image_paths, res_cls_split, cls_split


def read_class_dict_csv(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    result = {}
    for line in reader:
        if len(line) == 0:
            continue
        result[int(line[0])] = line[1]
    f.close()
    return result

    
def write_csv(image_label, csv_path):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        for key, value in image_label.items():
            writer.writerow([str(key), str(value)])
    pass

def filter_samples(image_paths, label_paths):
    img_xml = {}
    image_names = [os.path.basename(path).split('.')[0] for path in image_paths]
    label_names = [os.path.basename(path).split('.')[0] for path in label_paths]
    for i, image_name in enumerate(image_names):
        img_xml[image_paths[i]] = []
        for ii, lable_name in enumerate(label_names):
            if image_name == lable_name:
                img_xml[image_paths[i]].append(label_paths[ii])
    valid_img_xml = {k:v[0] for k, v in img_xml.items() if len(v)==1}
    unvalid_img_xml = {k:v for k, v in img_xml.items() if len(v)!=1}
    return valid_img_xml, unvalid_img_xml