# -*- coding: utf-8 -*-            
# @Author  : JinTian
# @Time    : 2024/3/15 13:01
# @Email   : 913178665@qq.com/tian.jin.zjcn@gmail.com
# @Tel     : +86-15657245055
# @WeChat  : tt5020772
# @File    : loss.py
# @Brief   : Such a fucking easy project, waste my life.
# 
# You can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.
import cv2
import numpy as np
import tensorflow as tf
from cfg.config import Config as C
from tensorflow.keras import losses

def cls_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred))

    return loss


def ctc_loss(labels, label_length, logits):
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1]) # 这里将产生一个数组[batchsize_length, sequense_length]
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=-1)

    return tf.reduce_mean(loss)

def iou_copmuter(x1, y1, w1, h1, x2, y2, w2, h2):
    # x1...:[b,32,64,5]
    xmin1 = x1 - 0.5 * w1
    xmax1 = x1 + 0.5 * w1
    ymin1 = y1 - 0.5 * h1
    ymax1 = y1 + 0.5 * h1

    xmin2 = x2 - 0.5 * w2
    xmax2 = x2 + 0.5 * w2
    ymin2 = y2 - 0.5 * h2
    ymax2 = y2 + 0.5 * h2

    # (xmin1,ymin1,xmax1,ymax1) (xmin2,ymin2,xmax2,ymax2)
    interw = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    interh = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interw * interh
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + 1e-6)
    # [b,32,64,5]
    return iou

def yolo_loss(detector_mask, matching_gt_boxes, matching_classes_oh, gt_boxes_grid, y_pred):
    # detector_mask:[b, 32, 64, 5, 1] 表示每个网格上哪个anchor为iou最大anchor
    # matching_gt_boxes:[b, 32, 64, 5, 5] x-y-w-h-l
    # matching_classes_oh:[b, 32, 64, 5, 2] l1-l2的概率
    # gt_boxes_grid:[b,4,5] 真实物体的标签 4表示物体最多有几个,5表示x-y-w-h-l
    # y_pred: [b, 32, 64, 5, 8] x-y-w-h-conf-l1-l2....
    pred_xy = y_pred[0]
    pred_wh = y_pred[1]
    pred_conf = y_pred[2]
    pred_box_class = y_pred[3]

    anchors = np.array(C.ANCHORS).reshape(5, 2)

    x = tf.range(C.GRID_W)
    y = tf.range(C.GRID_H)
    x, y = tf.meshgrid(x, y)
    xy_grid = tf.stack([x, y], axis=-1)
    xy_grid = tf.expand_dims(tf.expand_dims(xy_grid, axis=0), axis=3)
    xy_grid = tf.tile(xy_grid, [pred_xy.shape[0], 1,1,5,1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = pred_xy + xy_grid
    pred_wh = pred_wh * anchors

    n_detector_mask = tf.reduce_sum(tf.cast(detector_mask > 0., tf.float32)) #求出一共多少box在这个batch中

    xy_loss = detector_mask * tf.square(matching_gt_boxes[..., :2] - pred_xy) / (n_detector_mask + 1e-6)  #这里使用l2损失来计算中心点误差， 乘以detector_mask是保证有boxes的xy参与没有的不参与
    xy_loss = tf.reduce_sum(xy_loss)
    wh_loss = detector_mask * tf.square(tf.sqrt(matching_gt_boxes[..., 2:4]) - tf.sqrt(pred_wh)) / (n_detector_mask + 1e-6) #也是用l2损失来计算，依旧是有物体的box参与计算
    wh_loss = tf.reduce_sum(wh_loss)

    coord_loss = xy_loss + wh_loss

    class_loss = tf.keras.losses.CategoricalCrossentropy()(matching_classes_oh, pred_box_class)
    class_loss = tf.expand_dims(class_loss, -1) * detector_mask
    class_loss = tf.reduce_sum(class_loss) / (n_detector_mask + 1e-6)

    x1, y1, w1, h1 = matching_gt_boxes[..., 0], matching_gt_boxes[..., 1], matching_gt_boxes[..., 2], matching_gt_boxes[..., 3]
    x2, y2, w2, h2 = pred_xy[..., 0], pred_xy[..., 1], pred_wh[...,0],pred_wh[...,1]
    ious = iou_copmuter(x1, y1, w1, h1, x2, y2, w2, h2)
    ious = tf.expand_dims(ious, axis=-1)     # [b,32,64,5,1]

    pred_xy = tf.expand_dims(pred_xy, axis=4)
    pred_wh = tf.expand_dims(pred_wh, axis=4)
    pred_wh_half = pred_wh / 2.
    pred_xymin = pred_xy - pred_wh_half #这样得到了每个特征点上5个anchor的左上角
    pred_xymax = pred_xy + pred_wh_half #这样得到了每个特征点上5个anchor的右下角

    true_boxes_grid = tf.reshape(gt_boxes_grid, [gt_boxes_grid.shape[0], 1, 1, 1, gt_boxes_grid.shape[1], gt_boxes_grid.shape[2]])
    true_xy = true_boxes_grid[..., 0:2]
    true_wh = true_boxes_grid[..., 2:4]
    true_wh_half = true_wh / 2.
    true_xymin = true_xy - true_wh_half
    true_xymax = true_xy + true_wh_half

    intersectxymin = tf.maximum(pred_xymin, true_xymin)
    intersectxymax = tf.minimum(pred_xymax, true_xymax)
    intersect_wh = tf.maximum(intersectxymax - intersectxymin, 0.) # 这里是求解所有特征点上的所有anchor和真实目标框的交集高宽
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]   # 求出交并比面积

    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    union_area = pred_area + true_area - intersect_area
    iou_score = intersect_area / union_area  #大部分都是0
    best_iou = tf.reduce_max(iou_score, axis=4)
    best_iou = tf.expand_dims(best_iou, axis=-1)

    nonobj_detection = tf.cast(best_iou < 0.6, tf.float32)
    nonobj_mask = nonobj_detection * (1 - detector_mask)
    n_nonobj = tf.reduce_sum(tf.cast(nonobj_mask > 0., tf.float32))

    nonobj_loss = tf.reduce_sum(nonobj_mask * tf.square(-pred_conf)) / (n_nonobj + 1e-6)
    obj_loss = tf.reduce_sum(detector_mask * tf.square(ious - pred_conf)) / (n_detector_mask + 1e-6)

    loss = coord_loss + class_loss + nonobj_loss + 5 * obj_loss

    return loss, [nonobj_loss + 5 * obj_loss, class_loss, coord_loss]



def focal_loss(hm_pred, hm_true):
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-6, 1.)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-6, 1.)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss

def reg_l1_loss(y_pred, y_true, indices, mask):
    #-------------------------------------------------------------------------#
    #   获得batch_size和num_classes
    #-------------------------------------------------------------------------#
    b, c = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
    k = tf.shape(indices)[1]

    y_pred = tf.reshape(y_pred, (b, -1, c))
    length = tf.shape(y_pred)[1]
    indices = tf.cast(indices, tf.int32)

    #-------------------------------------------------------------------------#
    #   利用序号取出预测结果中，和真实框相同的特征点的部分
    #-------------------------------------------------------------------------#
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, k))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) +
                    tf.reshape(indices, [-1]))

    y_pred = tf.gather(tf.reshape(y_pred, [-1,c]),full_indices)
    y_pred = tf.reshape(y_pred, [b, -1, c])

    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    #-------------------------------------------------------------------------#
    #   求取l1损失值
    #-------------------------------------------------------------------------#
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)

    return reg_loss

def centernet_loss(args):
    #-----------------------------------------------------------------------------------------------------------------#
    #   hm_pred：热力图的预测值       (batch_size, 128, 128, num_classes)
    #   wh_pred：宽高的预测值         (batch_size, 128, 128, 2)
    #   reg_pred：中心坐标偏移预测值  (batch_size, 128, 128, 2)
    #   hm_true：热力图的真实值       (batch_size, 128, 128, num_classes)
    #   wh_true：宽高的真实值         (batch_size, max_objects, 2)
    #   reg_true：中心坐标偏移真实值  (batch_size, max_objects, 2)
    #   reg_mask：真实值的mask        (batch_size, max_objects)
    #   indices：真实值对应的坐标     (batch_size, max_objects)
    #-----------------------------------------------------------------------------------------------------------------#
    hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, indices = args
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    # total_loss = tf.Print(total_loss,[hm_loss,wh_loss,reg_loss])
    return total_loss



def arcface_loss(reallabs, embedding, weight, n_classes, margin=0.001, scale=12.0):
    embedding = tf.math.l2_normalize(embedding, axis=1)
    weight = tf.math.l2_normalize(weight, axis=0)

    # a·b = ||a||×||b||×cosθ
    # cosθ =  a·b /||a||×||b||
    # θ = arccos(a·b /||a||×||b||)
    cos_theta = tf.matmul(embedding, weight) # 英文a和b 已经归一化了，所以这里||a||×||b||等于1 cosθ =  a·b
    # theta = tf.acos(tf.clip_by_value(cos_theta, -1 + 1e-9, 1 - 1e-9))    # θ = arccos(a·b) 弧度制

    sin_theta = tf.sqrt(1.0 - tf.square(cos_theta) + 1e-7)
    cos_theta_m = cos_theta  * tf.cos(margin)  - sin_theta * tf.sin(margin) # f.cos(theta + margin) 三角恒等变换
    # cos_theta_m 这个值就相当于论文里头的cos(θ+m), 但是这个值只在正确项上有，非正确累没有，所以接下来肯定用到one-hot

    one_hot = tf.one_hot(reallabs, depth=n_classes)
    logits = (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * scale

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=reallabs, logits=logits))

    return loss




