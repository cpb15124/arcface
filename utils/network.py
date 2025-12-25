# -*- coding: utf-8 -*-            
# @Author  : JinTian
# @Time    : 2025/12/1 14:20
# @Email   : 913178665@qq.com/tian.jin.zjcn@gmail.com
# @Tel     : +86-15657255055
# @WeChat  : tt5020772
# @File    : network.py
# @Brief   : Such a fucking easy project, waste my life.
# 
# You can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import tensorflow as tf


class Network(object):
    def __init__(self, C):
        self.C = C

    def body(self, input_image):
        x = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', name="conv0", use_bias=False)(input_image)
        x = tf.keras.layers.BatchNormalization(name='bn0')(x)
        x = tf.keras.layers.Activation('silu')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', name="conv1", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x = tf.keras.layers.Activation('silu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', name="conv2", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name='bn2')(x)
        x = tf.keras.layers.Activation('silu')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name="conv3", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name='bn3')(x)
        x = tf.keras.layers.Activation('silu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', name="conv4", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name='bn4')(x)
        x = tf.keras.layers.Activation('silu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
        x = tf.keras.layers.Dense(self.C.EMBEDDING_SIZE, use_bias=False, name='embedding')(x)

        return x



