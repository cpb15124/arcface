# -*- coding: utf-8 -*-            
# @Author  : JinTian
# @Time    : 2025/12/7 16:05
# @Email   : 913178665@qq.com/tian.jin.zjcn@gmail.com
# @Tel     : +86-15657255055
# @WeChat  : tt5020772
# @File    : pb_demo.py
# @Brief   : Such a fucking easy project, waste my life.
# 
# You can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
import json
import random

import cv2 as cv
import numpy as np
import tensorflow as tf
from cfg.config import Config as C
import time


def l2_normalize(x, axis=1, epsilon=1e-12):
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
    norm = np.sqrt(np.maximum(square_sum, epsilon))  # 防止除以 0
    return x / norm


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
loaded_model = tf.saved_model.load("./logs/demo/")

print("所有签名函数：", list(loaded_model.signatures.keys()))  # 通常是 ['serving_default']

infer = loaded_model.signatures["serving_default"]

root = 'J:\\LLM\\'
dirs = os.listdir(root)

src1 = cv.imread('./imgs/Adam_Sandler_0001.jpg')
src1 = cv.resize(src1, (C.IMAGE_SIZE_W, C.IMAGE_SIZE_H))
src1 = src1 / 255.0

src2 = cv.imread('./imgs/Adam_Sandler_0003.jpg')
src2 = cv.resize(src2, (C.IMAGE_SIZE_W, C.IMAGE_SIZE_H))
src2 = src2 / 255.0

src3 = cv.imread('./imgs/Abdullah_Ahmad_Badawi_0001.jpg')
src3 = cv.resize(src3, (C.IMAGE_SIZE_W, C.IMAGE_SIZE_H))
src3 = src3 / 255.0


input1 = tf.convert_to_tensor(src1[np.newaxis, :, :, :], dtype=tf.float32)
input2 = tf.convert_to_tensor(src2[np.newaxis, :, :, :], dtype=tf.float32)
input3 = tf.convert_to_tensor(src3[np.newaxis, :, :, :], dtype=tf.float32)
result1 = infer(input1)
result2 = infer(input2)
result3 = infer(input3)
pred1 = result1["embedding"].numpy()
pred2 = result2["embedding"].numpy()
pred3= result3["embedding"].numpy()
embedding1 = l2_normalize(pred1)[0]
embedding2 = l2_normalize(pred2)[0]
embedding3 = l2_normalize(pred3)[0]

cosine_similarity1 = np.dot(embedding1, embedding2)
cosine_similarity2 = np.dot(embedding1, embedding3)

print("余弦相似度：", cosine_similarity1)
if cosine_similarity1 > 0.5:  # ArcFace 常用阈值区间：0.45 ~ 0.6
    print("图1和图2判断结果：同一个人")
else:
    print("图1和图2判断结果：不是同一个人")

print("余弦相似度：", cosine_similarity2)
if cosine_similarity2 > 0.5:  # ArcFace 常用阈值区间：0.45 ~ 0.6
    print("图1和图3判断结果：同一个人")
else:
    print("图1和图3判断结果：不是同一个人")
