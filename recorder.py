#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
"""
@author: red0orange
@file: recorder.py
@time:  7:32 PM
@desc:
"""
import traceback
from runx.logx import LogX
import shutil
import os
import torch


class Recorder(LogX):
    def __init__(self):
        super(Recorder, self).__init__()
        self.record_dir = None
        self.save_last = None

        self._init = False
        pass

    def init(self, save_root, save_modules=[]):
        # init recorder
        self.initialize(record_dir=save_root, save_last=5, modules_need_to_save=save_modules,
                        coolname=True, tensorboard=True, eager_flush=True)
        self._init = True
        pass

    def initialize(self, record_dir=None, save_last=5, modules_need_to_save=[], coolname=False, hparams=None,
                   tensorboard=False, no_timestamp=False, global_rank=0,
                   eager_flush=True):
        # 调用runx中的初始化，保存路径为record_dir/metrics
        super(Recorder, self).initialize(logdir=os.path.join(record_dir, 'metrics'), coolname=coolname, hparams=hparams,
                                         tensorboard=tensorboard, no_timestamp=no_timestamp, global_rank=global_rank,
                                         eager_flush=eager_flush)
        self.record_dir = record_dir
        self.save_last = save_last

        # 创建需要的子文件夹
        os.makedirs(os.path.join(self.record_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.record_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(self.record_dir, 'code'), exist_ok=True)

        # 保存传入的参数中需要保存的模块文件
        file_save_root = os.path.join(self.record_dir, 'code')
        file_paths = []
        for module in modules_need_to_save:
            try:
                file_path = module.__file__
            except Exception:
                raise BaseException('error load module file to save')
            file_paths.append(file_path)
        # 调用这个函数的主文件默认保存
        file_paths.append(traceback.extract_stack()[0].filename)
        file_paths = [os.path.abspath(i) for i in file_paths]
        file_paths = list(set(file_paths))
        for file_path in file_paths:
            new_path = os.path.join(file_save_root, os.path.basename(file_path))
            shutil.copy(file_path, new_path)
        pass

    def save_model(self, save_dict, epoch, best=False):
        if not self._init:
            raise BaseException("not init")
        # 使用自己的保存方法
        save_root = os.path.join(self.record_dir, 'models')
        save_path = os.path.join(save_root, f'{epoch}.pth')
        torch.save(save_dict, save_path)
        if not best:
            if os.path.exists(os.path.join(save_root, f'{epoch-self.save_last}.pth')):
                os.remove(os.path.join(save_root, f'{epoch-self.save_last}.pth'))
        pass


recorder = Recorder()
