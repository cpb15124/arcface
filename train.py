# -*- coding: utf-8 -*-            
# @Author  : JinTian
# @Time    : 2025/12/1 14:19
# @Email   : 913178665@qq.com/tian.jin.zjcn@gmail.com
# @Tel     : +86-15657255055
# @WeChat  : tt5020772
# @File    : train.py
# @Brief   : Such a fucking easy project, waste my life.
# 
# You can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
import tensorflow as tf
from tensorflow.keras import layers
from utils.network import Network
from utils.arcfacelayer import ArcFaceLayer
from utils.dataset import Dataset
from cfg.config import Config as C
from cfg.args import parse_args
from utils import loss
from datetime import datetime

class ARCFACE(object):
    def __init__(self, args):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu = gpus[0]
            tf.config.experimental.set_memory_growth(gpu, True)  # 设置GPU0显存动态分配
        self.args = args
        self.network = Network(C)
        self.arcfacelayer = ArcFaceLayer(C.EMBEDDING_SIZE, C.NUM_CLASSES)
        self.dataset = Dataset(args)
        self.loss = loss

    def train(self):
        input_image = layers.Input((C.IMAGE_SIZE_H, C.IMAGE_SIZE_W, 3), dtype='float32', name='input')
        input_labels = layers.Input(shape=(), dtype=tf.int32, name='labels')

        embedding = self.network.body(input_image)
        logits = self.arcfacelayer(embedding, input_labels)

        train_model = tf.keras.Model(inputs=[input_image, input_labels], outputs=logits, name='train_model')
        train_model.summary()
        infer_model = tf.keras.Model(inputs=input_image, outputs=embedding, name='infer_model')
        infer_model.summary()

        try:
            print('=> Restoring weights from: %s ... ' % self.args.initial_weight)
            train_model.load_weights(self.args.initial_weight)
            print("done.")
        except:
            print('=> %s does not exist !!!' % self.args.initial_weight)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=C.LEARNING_RATE, momentum=0.9, nesterov=True)

        image_label = self.dataset.get_file()
        for iter in range(C.ITERATION):
            images, labels = self.dataset.get_data(C, image_label)
            with tf.GradientTape() as tape:
                y_pred = train_model([images, labels], training=True)
                cls_total_loss = self.loss.cls_loss(labels, y_pred)

            grads = tape.gradient(cls_total_loss, train_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, train_model.trainable_variables))

            if iter % C.ITER_SUMMARY == 0 or (iter + 1) == C.ITERATION:
                print('summary_iter: %d, learn_rate: %f, model_loss: %.5f' % (iter, C.LEARNING_RATE, cls_total_loss))

            if iter % C.ITER_SAVE == 0 or (iter + 1) == C.ITERATION:
                timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
                save_loss = "{:.3f}".format(cls_total_loss)
                filename = "weights_loss_{}_date_{}.h5".format(save_loss, timestamp)
                train_model.save_weights(self.args.logs + filename)
                print("Saved model weights to:", filename)
                infer_model.save(self.args.demo_weight)
                print("pb model saved as well", )



if __name__ == '__main__':
    args = parse_args()
    af = ARCFACE(args)
    af.train()