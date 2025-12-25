# -*- coding: utf-8 -*-            
# @Author  : JinTian
# @Time    : 2025/12/1 14:20
# @Email   : 913178665@qq.com/tian.jin.zjcn@gmail.com
# @Tel     : +86-15657255055
# @WeChat  : tt5020772
# @File    : dataset.py
# @Brief   : Such a fucking easy project, waste my life.
# 
# You can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
import random
import cv2 as cv
import numpy as np

class Dataset(object):
    def __init__(self, args):
        self.args = args
        self.labeldic = {}


    def get_file(self):
        files = os.listdir(self.args.train_data)
        img_lab = []
        for file in files:
            if file not in self.labeldic:
                self.labeldic[file] = len(self.labeldic)
            fatherpath = self.args.train_data + '/' + file
            imgs = os.listdir(fatherpath)
            for img in imgs:
                imgdir = fatherpath + '/' + img
                img_lab.append([imgdir, self.labeldic[file]])
        # random.seed(200)
        random.shuffle(img_lab)
        return img_lab


    def get_data(self, C, image_label):
        bs = random.sample(image_label, C.BATCH_SIZE)
        ######数据处理和增强######
        images = np.zeros((C.BATCH_SIZE, C.IMAGE_SIZE_H, C.IMAGE_SIZE_W, 3), dtype=np.float)
        labels = np.zeros((C.BATCH_SIZE), dtype=np.int)

        for i in range(len(bs)):
            imgpath = bs[i][0]
            label = bs[i][1]
            #####image######
            # src = cv.imread(imgpath)
            with open(imgpath, 'rb') as f:
                chunk = f.read()
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            src = cv.imdecode(chunk_arr, cv.IMREAD_COLOR)
            h, w = src.shape[:2]
            side_length = max(h, w)
            padded = np.zeros((side_length, side_length, src.shape[2]), dtype=np.uint8)
            top = (side_length - h) // 2
            left = (side_length - w) // 2
            padded[top:top + h, left:left + w] = src
            src = padded

            aug_functions = [
                self.random_cropping,
                self.random_occlusion,
                self.gaussian_blur,
                self.random_grayscale,
                self.color_jitter,
                # 还可以继续加：self.random_rotate, self.random_brightness, etc.
            ]
            num_augs = random.randint(1, min(4, len(aug_functions)))
            selected_augs = random.sample(aug_functions, num_augs)
            random.shuffle(selected_augs)
            for aug_func in selected_augs:
                src = aug_func(src)

            img = cv.resize(src, (C.IMAGE_SIZE_W, C.IMAGE_SIZE_H))
            imgnormalized = img / 255.0
            images[i] = imgnormalized
            #####label######
            labels[i] = label

        # one_hot_labels = np.eye(len(self.labeldic))[labels]

        return images, labels # one_hot_labels


    def random_cropping(self, image, min_ratio=0.8, max_ratio=1.0):
        h, w = image.shape[:2]
        ratio = random.random()
        scale = min_ratio + ratio * (max_ratio - min_ratio)
        new_h = int(h * scale)
        new_w = int(w * scale)
        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)

        image = image[y:y + new_h, x:x + new_w, :]

        return image

    def random_occlusion(self, image, min_ratio=0.02, max_ratio=0.25, color_mode='mean'):
        h, w = image.shape[:2]
        img_area = h * w

        # 2. 随机决定遮挡面积（在 min_ratio ~ max_ratio 之间）
        mask_area = random.uniform(min_ratio, max_ratio) * img_area
        # 保证宽高比在 0.3~3 之间（避免太细长）
        aspect_ratio = random.uniform(0.3, 3.0)

        mask_h = int(np.sqrt(mask_area / aspect_ratio))
        mask_w = int(aspect_ratio * mask_h)

        # 防止超出边界
        if mask_h > h or mask_w > w:
            mask_h = min(mask_h, h)
            mask_w = min(mask_w, w)

        # 3. 随机选择遮挡位置
        top = random.randint(0, h - mask_h)
        left = random.randint(0, w - mask_w)

        # 4. 填充方式（四选一）
        color_mode = random.choice(['mean', 'random'])
        if color_mode == 'mean':  # 最常用！模拟自然遮挡
            mask_color = image.mean(axis=(0, 1)).astype(np.uint8)  # 整图均值
        if color_mode == 'random':  # 随机噪声
            mask_color = np.random.randint(0, 256, size=3, dtype=np.uint8)

        # 5. 执行遮挡
        image[top:top + mask_h, left:left + mask_w] = mask_color

        return image

    def gaussian_blur(self, image, max_ksize=7):
        ksize = random.choice([3, 5, max_ksize if max_ksize % 2 == 1 else max_ksize - 1])
        image = cv.GaussianBlur(image, (ksize, ksize), sigmaX=0)

        return image

    def random_grayscale(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # 变回3通道

        return image

    def color_jitter(self, image,
                     brightness=0.4,  # 亮度 ±40%
                     contrast=0.4,  # 对比度 ±40%
                     saturation=0.4,  # 饱和度 ±40%
                     hue=0.1):  # 色相 ±10%（HSV空间）
        b = random.uniform(1 - brightness, 1 + brightness)
        c = random.uniform(1 - contrast, 1 + contrast)
        s = random.uniform(1 - saturation, 1 + saturation)
        h = random.uniform(-hue, hue) * 180  # OpenCV hue范围是0~180

        # 1. 亮度 + 对比度（在BGR空间）
        image = cv.convertScaleAbs(image, alpha=c, beta=(b - 1) * 127)  # beta控制亮度偏移

        # 2. 饱和度 + 色相（转到HSV空间）
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)

        # 饱和度通道（S）乘系数
        hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)

        # 色相通道（H）加偏移
        hsv[..., 0] = (hsv[..., 0] + h) % 180

        image = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

        return image

if __name__ == '__main__':
    from cfg.args import parse_args
    from cfg.config import Config as C

    args = parse_args()
    D = Dataset(args)
    image_label = D.get_file()
    images, labels = D.get_data(C, image_label)
    print(1)
