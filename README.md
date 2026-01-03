# ArcFace Tutorial / ArcFace 教程

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0%2B-orange)](https://www.tensorflow.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-green)]()

The most minimal ArcFace face comparison tutorial project.  
_本项目是最精简的 ArcFace 人脸比对教学项目。_

This project removes all unnecessary complexity, retaining only the core implementation to help beginners quickly understand and implement ArcFace.  
_本项目去除了所有不必要的复杂依赖和代码，只保留核心实现，让初学者能在最短时间内理解 ArcFace 的原理并上手实践。_

## Teaching Video ## 教学视频

Watch the detailed step-by-step video tutorial (Chinese with code explanation):  
观看详细的逐行代码讲解教学视频（中文讲解）：

[📺 Bilibili 教学视频：最精简 ArcFace 人脸比对从零实现](https://www.bilibili.com/video/BV1BvvZB7ErZ/?spm_id_from=333.1387.homepage.video_card.click&vd_source=3c72f427c89ff1a1636e470f09ea3c76)

Highly recommended to watch the video first, then run the code in this repository for best learning experience!  
强烈建议先观看视频讲解，再结合本仓库代码运行，学习效果最佳！

## Requirements / 环境要求
- Python 3.6 or higher / Python 3.6 或更高版本
- TensorFlow-gpu 2.6.0 or higher / TensorFlow-gpu 2.6.0 或更高版本
- OpenCV 4.9.0.80 or other versions / OpenCV 4.9.0.80 或其它版本均可
- NumPy 1.21.5 or other versions / NumPy 1.21.5 或其它版本均可
- OS Support: Windows / Linux (macOS not tested, theoretically compatible) / 系统支持：Windows / Linux（macOS 未测试，理论兼容）

## Quick Start / 快速开始
### 1. Clone the repository / 1. 克隆仓库
```bash
git clone https://github.com/cpb15124/arcface.git
cd arcface
```
### 2. Install dependencies / 2. 安装依赖
```bash
pip install -r requirements.txt
```
### 3. Download dataset(optional, e.g., LFW) / 3. 下载数据集（可选，例如LFW）
```bash
https://pan.baidu.com/s/1tr-evZMgUpMgvfE_I9lMYw
code: f3zc
```
### 4. Train the model / 4. 训练模型
```bash
python train.py --train_data <your_dataset_path>
```


