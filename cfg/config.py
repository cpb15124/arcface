# -*- coding: utf-8 -*-            
# @Author  : JinTian
# @Time    : 2025/12/1 14:27
# @Email   : 913178665@qq.com/tian.jin.zjcn@gmail.com
# @Tel     : +86-15657255055
# @WeChat  : tt5020772
# @File    : config.py
# @Brief   : Such a fucking easy project, waste my life.
# 
# You can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
from cfg.args import parse_args

args = parse_args()


def get_output_unit():
    dirs = os.listdir(args.train_data)

    return len(dirs)

class Config(object):
    BATCH_SIZE = 64

    NUM_CLASSES = get_output_unit()

    IMAGE_SIZE_H = 64

    IMAGE_SIZE_W = 64

    EMBEDDING_SIZE = 256

    LEARNING_RATE = 0.0001

    ITERATION = 50000

    ITER_SUMMARY = 20

    ITER_SAVE = 400

if __name__ == '__main__':
    c = Config()
    print(c.NUM_CLASSES)