# -*- coding: utf-8 -*-            
# @Author  : JinTian
# @Time    : 2025/12/5 14:59
# @Email   : 913178665@qq.com/tian.jin.zjcn@gmail.com
# @Tel     : +86-15657255055
# @WeChat  : tt5020772
# @File    : arcfacelayer.py
# @Brief   : Such a fucking easy project, waste my life.
# 
# You can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math
import tensorflow as tf

class ArcFaceLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

        self.cos_m = tf.constant(math.cos(self.margin), dtype=tf.float32)
        self.sin_m = tf.constant(math.sin(self.margin), dtype=tf.float32)
        self.th = tf.constant(math.cos(math.pi - self.margin), dtype=tf.float32)
        self.mm = tf.constant(math.sin(self.margin) * self.margin, dtype=tf.float32)  # 直接算好！

    def build(self, input_shape):
        self.w = self.add_weight(
            name='weights',
            shape=[int(input_shape[-1]), self.num_classes],
            initializer='glorot_uniform',
            trainable=True,
            regularizer=tf.keras.regularizers.l2(5e-4)
        )


    def call(self, embeddings, labels):
        normed_embds = tf.nn.l2_normalize(embeddings, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        cos_t = tf.clip_by_value(cos_t, -1.0 + 1e-6, 1.0 - 1e-6)

        sin_t = tf.sqrt(tf.clip_by_value(1.0 - tf.square(cos_t), 1e-6, 1.0), name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logits = tf.where(mask == 1., cos_mt, cos_t)
        logits = logits * self.logist_scale

        return logits


