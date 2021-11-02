#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
"""
@author: red0orange
@file: args.py
@time:  11:10 PM
@desc:
"""
import argparse
import ast
import sys
import os
import time
import numpy as np
import torch
import shutil
import random


class SingleModel(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '__instance__'):
            cls.__instance__ = super(SingleModel, cls).__new__(cls)
        return cls.__instance__

    def init(self):
        pass


def get_class_attr(c):
    name_list = [i for i in dir(c) if not i.startswith('__')]
    result = {k: getattr(c, k) for k in name_list}
    return result


class BaseOption(object):
    class Arg:
        custom_data = True
        pass

    class Help:
        outcome_root = "实验结果保存的根目录"
        gpu = "选择使用的gpu"
        seed = "设置使用的随机种子"

    class ExtraArg:
        outcome_root = 'outcomes'
        gpu = "0"
        seed = 28
        description = ''
        name = ''


class Args(SingleModel):
    def init(self, option, expand_mode=False, code_mode=False):
        self.expand_mode = expand_mode
        self.code_mode = code_mode
        self.parse_args(option)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        if self.name == '':
            self.model = '' if not hasattr(self, 'model') else self.self.model
            self.name = self.model + time.strftime('(%m-%d_%H:%M)', time.localtime(time.time()))
        else:
            self.name += time.strftime('(%m-%d_%H:%M)', time.localtime(time.time()))

        # 根据参数中的outcome_root、description、name的得到结果的保存文件夹路径
        if self.description == '':
            self.save_root = os.path.join(self.outcome_root, self.name)
        else:
            self.save_root = os.path.join(self.outcome_root, self.description, self.name)
        # 若本身已存在这个路径，就删除（一般情况下不会删除，因为有加时间戳），然后再创建这个文件夹
        if os.path.exists(self.save_root):
            shutil.rmtree(self.save_root)
        os.makedirs(self.save_root)

        return self

    def parse_args(self, option):
        # 跑自己代码的两种模式
        # 模式一：代码模式，一般直接在IDE中运行，不想要外部设置参数
        # 模式二：外部输入模式，一般在终端运行代码，需要外部输入设置参数
        # 跑开源代码的模式
        # 一般开源代码都是封装成parser，然后一堆需要外部输入的参数，这时候外部输入是正常工作的，不用修改，主要是
        # 代码模式要利用这个工具回复正常，即可以直接很方便地在代码里面设置参数而不需要去更改它parser里面的默认值
        # ，这个模式定义为拓展模式，在拓展模式下就是想要在开源代码中使用代码模式，直接在代码中设置参数
        expand_mode = self.expand_mode  # 设置为拓展模式，模拟外部输入参数运行开源代码
        code_mode   = self.code_mode    # 设置为代码模式，不从外部读取参数

        class_attr = get_class_attr(option)
        parse_args = get_class_attr(class_attr["Arg"])
        parse_help = get_class_attr(class_attr["Help"])
        parse_extra_args = get_class_attr(class_attr["ExtraArg"])

        parse_extra_args_names = list(parse_extra_args.keys())
        assert "outcome_root" in parse_extra_args_names, "必须设置实验结果保存的根目录"

        if expand_mode:
            for arg_name, arg_value in parse_args.items():
                sys.argv.extend(["--{}".format(arg_name), "{}".format(arg_value)])
            args_ = parse_args
        else:
            if not code_mode:
                parser = argparse.ArgumentParser()
                for arg_name in parse_args.keys():
                    arg_type = type(parse_args[arg_name])
                    arg_default_value = parse_args[arg_name]
                    arg_help = "" if parse_help.get(arg_name) is None else str(parse_help.get(arg_name))
                    if arg_type == bool:
                        parser.add_argument(f'--{arg_name}', type=ast.literal_eval, default=arg_default_value,
                                            help=arg_help)
                    else:
                        parser.add_argument(f'--{arg_name}', type=arg_type, default=arg_default_value, help=arg_help)
                args_ = vars(parser.parse_args())
            else:
                args_ = parse_args

        inter_args = [i for i in list(args_.keys()) if i in list(parse_extra_args.keys())]
        assert len(inter_args) == 0, "普通参数与额外参数不能重复"

        for arg_name, arg_value in {**args_, **parse_extra_args}.items():
            setattr(self, arg_name, arg_value)
            pass

    def get_args(self):
        return {k: getattr(self, k) for k in self.__dict__.keys() if not k.startswith(('_', '__'))}


option = Args()

if __name__ == '__main__':
    option.init(BaseOption, expand_mode=True, code_mode=True)

    print('==========================================')
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--custom_data', type=str, default="0, 1", help="")
    args_ = vars(parser.parse_args())
    for key, value in args_.items():
        print(key, value)
    print('==========================================')
    for value in sys.argv:
        print(value)
    print('==========================================')
    for key, value in get_class_attr(option).items():
        print(key, value)
    pass
