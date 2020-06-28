# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy.linalg import norm

SZ = 20  # 训练图片长宽
# 来自opencv的sample，用于svm训练 对一个图像进行抗扭斜处理
def deskew(img):
    """
            :param img:要进行扭曲处理的图片
            :return:img：处理后图片
    """
    m = cv2.moments(img)

    # 图像的中心矩 可以帮助我们计算面积 m包括了很多轮廓信息
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    # 其中src - 输入图像。 M - 变换矩阵。 dsize - 输出图像的大小。 flags - 插值方法的组合（int  类型！）
    # INTER_LINEAR 是线性插值算法  WARP_INVERSE_MAP标志位，反变换
    return img


#  hog方向梯度直方图 梯度方向直方图(HOG)
def preprocess_hog(digits):
    """
            :param digits:要进行扭曲处理的图片
            :return:np.float32(samples)：直方图数据
    """
    samples = []
    for img in digits:
        # cv2.imshow("namee", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        # 将图像划分为4个小的方块 对每个小方块计算他们的朝向直方图
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        # 每个子块给你一个包含16值的向量，4个子块就是64位
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        # ravel 将多维数组转换为一维数组   bincount首先是通过所引数求出加权值
        hist = np.hstack(hists)
        #4 个小方块的 4 个向量就组成了这个图像的特征向量（包含 64 个成员）。这就是我们要训练数据的特征向量。
        
        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        # 实现数据归一化
        # 求范数
        # print("hist",hist)
        samples.append(hist)
    return np.float32(samples)


provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "青",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]



