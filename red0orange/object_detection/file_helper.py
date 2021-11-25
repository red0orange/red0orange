import numpy as np
import xml.dom.minidom as minidom


def parse_xml_to_boxes(xml_path):
    """针对特定格式保存目标检测标注数据的xml，解析为box的array

    Args:
        xml_path (str): 

    Returns:
        numpy array: 
        list:
    """
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


def parse_txt_to_boxes(txt_path):
    """针对特定格式保存目标检测标注数据的txt，解析为box的array

    Args:
        txt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    data = np.loadtxt(txt_path)
    if len(data.shape) == 1:
        if not data.shape[0]:
            return np.array([])
        data = data[None, ...]
    data[:, 1] -= (data[:, 3] / 2)
    data[:, 2] -= (data[:, 4] / 2)
    data[:, 3] += data[:, 1]
    data[:, 4] += data[:, 2]
    return data

def parse_txt_to_array(txt_path):
    """解析txt得到array，与np.loadtxt的主要区别在于针对空文件返回的是空数组而不是None

    Args:
        txt_path (str): 

    Returns:
        numpy array: 
    """
    data = np.loadtxt(txt_path)
    if len(data.shape) == 1:
        if not data.shape[0]:
            return np.array([])
        data = data[None, ...]
    return data