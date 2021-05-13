# encoding: utf-8
"""
@author: red0orange
@file: file.py
@desc: 文件相关的一些通用utils
"""
import os
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
