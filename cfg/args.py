# -*- coding: utf-8 -*-            
# @Author  : JinTian
# @Time    : 2025/12/1 14:29
# @Email   : 913178665@qq.com/tian.jin.zjcn@gmail.com
# @Tel     : +86-15657255055
# @WeChat  : tt5020772
# @File    : args.py
# @Brief   : Such a fucking easy project, waste my life.
# 
# You can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ARCFACE')

    parser.add_argument('--train_data_path', dest='train_data', help="train dataset location",
                        default='J:\\LLM\\', type=str)

    parser.add_argument('--logs', dest='logs', help="events logs files saveing path",
                        default='./logs/', type=str)

    parser.add_argument('--initial_weight', dest='initial_weight', help="initial weight for ckpt",
                        default='./logs/pretrain/weights_loss_0.191_date_20251225-12-03-08.h5', type=str)

    parser.add_argument('--demo_weight', dest='demo_weight', help="initial weight for ckpt",
                        default='./logs/demo/', type=str)

    args = parser.parse_args()

    return args

