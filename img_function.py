# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import img_math
import img_recognition
import config

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
PROVINCE_START = 1000


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    # ML 机器学习库 核心
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)#设置核函数
        self.model.setC(C)#设置惩罚项
        self.model.setKernel(cv2.ml.SVM_RBF)#高斯核 适用于非线性数据
        self.model.setType(cv2.ml.SVM_C_SVC)#允许类别的不完美分离与罚分乘数C的异常值

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    #    cv2.ml.ROW_SAMPLE  每个训练样本都是一行样本

    # 预测
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()
#     ravel() 将多维转换为一维

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
class CardPredictor:
    def cv_show(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __init__(self):
        pass

    def __del__(self):
        self.save_traindata()



    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # c是惩罚系数，即对误差宽容度，越大越不能容忍错误
        # gamma选择RBF函数作为kernel 越大支持向量机越少
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:

            chars_train = []
            chars_label = []
            # 喂数据
            for dirpath, dirnames, filenames in os.walk("train\\chars2"):
                # dirpathString 目录路径 dirnames文件夹下的子文件夹list 包含dirpath下全部目录  files遍历文件中的文件集合list非目录文件
                if len(os.path.basename(dirpath)) > 1:

                    continue
                root_int = ord(os.path.basename(dirpath))
                # 返回十进制数字
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    # 把目录和文件合成一个路径
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(root_int)


            chars_train = list(map(img_recognition.deskew, chars_train))
            # 根据函数映射图片最后以列表返回
            chars_train = img_recognition.preprocess_hog(chars_train)
            # 得到图片的直方图 作为特征图像
            chars_label = np.array(chars_label)
            self.model.train(chars_train, chars_label)

        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for dirpath, dirnames, filenames in os.walk("train\\charsChinese"):
                if not os.path.basename(dirpath).startswith("zh_"):
                    continue
                pinyin = os.path.basename(dirpath)
                index = img_recognition.provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(index)
            chars_train = list(map(img_recognition.deskew, chars_train))
            chars_train = img_recognition.preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            self.modelchinese.train(chars_train, chars_label)

    # 保存训练集
    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")

    # 图像第一次预处理
    def img_first_pre(self, car_pic_file):
        """
        :param car_pic_file: 图像文件
        :return:已经处理好的图像文件 原图像文件
        """
        if type(car_pic_file) == type(""):
            img = img_math.img_read(car_pic_file)
        else:
            img = car_pic_file

        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
            # cv_show('缩小的图片', img)

        blur =3
        img = cv2.GaussianBlur(img, (blur,blur), 0)
        # cv_show('Gauss', img)
        oldimg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv_show('BGR2GRAY', img)
        Matrix = np.ones((20, 20), np.uint8)
        # 根据给定类型返回一个矩阵
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
        # cv_show('open', img_opening)
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
        # cv_show('重叠', img_opening)
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv_show("二值化",img_thresh);
        img_edge = cv2.Canny(img_thresh, 100, 200)
        # cv_show('边缘检测', img_edge)
        Matrix = np.ones((4, 19), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        # cv_show('形态学操作', img_edge2)
        return img_edge2, oldimg

    # 颜色加轮廓识别
    def img_color_contours(self, img_contours, oldimg):
        """
        :param img_contours: 预处理好的图像
        :param oldimg: 原图像
        :return: 已经定位好的车牌
        """
        # cv_show("s",img_contours)
        if img_contours.any():
            config.set_name(img_contours)
        pic_hight, pic_width = img_contours.shape[:2]
        card_contours = img_math.img_findContours(img_contours)
        # 进行校正 仿射
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)
        colors, car_imgs = img_math.img_color(card_imgs)
        predict_result = []
        card_color = None

        # 通过波峰找到文字区
        for index, color in enumerate(colors):
            if color in ("blue", "yello", "green"):
                card_img = card_imgs[index]
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                x_histogram = np.sum(gray_img, axis=1)
                # axis=1列压缩
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = img_math.find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                # cv_show('截取图片', gray_img)
                row_num, col_num = gray_img.shape[:2]
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = img_math.find_waves(y_threshold, y_histogram)
                if len(wave_peaks) <= 6:
                    print("peak less 1:", len(wave_peaks))
                    continue
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]

                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 or wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0 #分割距离
                for index, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if index > 0:
                    wave = (wave_peaks[0][0], wave_peaks[index][1])
                    wave_peaks = wave_peaks[index + 1:]
                    wave_peaks.insert(0, wave)
                point = wave_peaks[2]

                # 车牌前两位 和后面之间的点
                point_img = gray_img[:, point[0]:point[1]]
                # 截取一部分
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)
                # 去除圆点

                # print("wave_peak", wave_peaks)
                if len(wave_peaks) <= 6:
                    print("peak less 2:", len(wave_peaks))
                    continue
                if len(wave_peaks) > 8:
                    wave_peaks.pop(len(wave_peaks)-1)
                # 分离车牌字符
                part_cards = img_math.seperate_card(gray_img, wave_peaks)
                for i, part_card in enumerate(part_cards):
                    # enumerate 遍历数据对象组成索引序列
                    # 可能是固定车牌的铆钉

                    if np.mean(part_card) < 255 / 5:
                        print("a point")
                        continue
                    part_card_old = part_card
                    w = abs(part_card.shape[1] - SZ) // 2
                    # 训练图片长宽
                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    # 添加边框 参数：1 要处理的原图 2 上下左右要扩展的像素 6 边框类型 通过给定的属性添加颜色边框
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    # 图片缩小的方式
                    part_card = img_recognition.preprocess_hog([part_card])
                    if i == 0:
                        # 模型预测,输入测试集,输出预测结果
                        resp = self.modelchinese.predict(part_card)
                        print("chinese",resp)
                        charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
                        print("汉字", charactor)
                    else:
                        resp = self.model.predict(part_card)

                        charactor = chr(resp[0])
                        print("字母数字", charactor)
                    #     chr 返回对应字符
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 9:  # 1太细，有可能认为是边缘
                            continue
                    predict_result.append(charactor)
                # 车牌信息
                roi = card_img
                # 车牌图片
                card_color = color
                # 颜色
                break
        return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色
    # 只使用颜色识别
    def img_only_color(self, filename, oldimg, img_contours):
        """
        :param filename: 图像文件
        :param oldimg: 原图像文件
        :return: 已经定位好的车牌
        """
        # cv_show("a",filename)
        pic_hight, pic_width = img_contours.shape[:2]
        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([60, 60, 60])
        upper_green = np.array([108, 255, 255])
        # 定义颜色的上下线范围
        hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # cv2.inRange 1原图 2 上下限 当大于或者小于的时候就取零 其他取 255
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # 对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1 & 1 = 1，1 & 0 = 0，0 & 1 = 0，0 & 0 = 0
        # 根据阈值找到对应颜色
        # cv_show("1",output)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        Matrix = np.ones((10, 10), np.uint8)
        img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)
        # cv_show("1",img_edge1)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        # cv_show("1", img_edge2)
        # 先闭后开操作 一开始太大导致图片混为一起

        card_contours = img_math.img_findContours(img_edge2)
        # 外接轮廓
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)

        # 进行校正仿射
        colors, car_imgs = img_math.img_color(card_imgs)
        # 车牌颜色判别

        predict_result = []
        # 字符
        roi = None
        # 颜色图片
        card_color = None
        # 颜色

        for i, color in enumerate(colors):

            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]
                # cv_show("ss", card_img)
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                    # cv_show("ss",gray_img)
                # .bitwise_not白色变黑色
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # 自动计算阈值
                # 查找水平波峰
                x_histogram = np.sum(gray_img, axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = img_math.find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                # cv_show('f-9', gray_img)
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                # cv_show('f-9', gray_img)
                # 按比例缩小
                y_histogram = np.sum(gray_img, axis=0)
                # 把列归到一起
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = img_math.find_waves(y_threshold, y_histogram)
                # 字符数应该大于6
                if len(wave_peaks) < 6:
                    print("peak less 1:", len(wave_peaks))
                    continue
                # print("wave_peak1", wave_peaks)
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 or wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)
                # print("wave_peak1", wave_peaks)
                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)
                # 删除中间的圆点
                # print("wave_peak", wave_peaks)
                if len(wave_peaks) <= 6:
                    print("peak less 2:", len(wave_peaks))
                    continue

                # 分离车牌字符
                part_cards = img_math.seperate_card(gray_img, wave_peaks)

                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉

                    if np.mean(part_card) < 255 / 5:
                        print("a point")
                        continue
                    part_card_old = part_card

                    w = abs(part_card.shape[1] - SZ) // 2

                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    # 添加边框 参数：1 要处理的原图 2 上下左右要扩展的像素 6 边框类型 通过给定的属性添加颜色边框
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    # 图片缩小
                    part_card = img_recognition.preprocess_hog([part_card])

                    # 训练用

                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(resp[0])
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 9:  # 1太细，认为是边缘
                            continue
                    predict_result.append(charactor)

                roi = card_img
                card_color = color
                break
        return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色

