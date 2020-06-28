# -*- coding: utf-8 -*-

import cv2
import numpy as np

Min_Area = 2000

"""
该文件包含读文件函数
取零值函数
矩阵校正函数
颜色判断函数
"""


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 图片读取
def img_read(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    # 以uint8方式读取filename 放入imdecode中，cv2.IMREAD_COLOR读取彩色照片


# 点界限 不让点小于零
def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

# 精确定位
def accurate_place(card_img_hsv, limitMin, limitMax, color):
    """
        :param card_img_hsv: HSV颜色模型的图片
        :param limitMin limitMax: 颜色取值范围
        :param color: 颜色
        :return:xl, xr, yh, yl：返回车牌的四个点 带l就是起始左上，xr宽、yh高
    """
    # cv_show("hsv",card_img_hsv)
    # card_img_hsv  hsv 是更容易划分颜色范围的  limit1就是min  limit2就是max
    row_num, col_num = card_img_hsv.shape[:2]
    # 转化为二维数组 分行列 行列就是图片宽高

    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    row_num_limit = 21
    # 设置车牌长
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
    # 车牌宽
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limitMin < H <= limitMax and 34 < S and 46 < V:
                # 筛选颜色
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limitMin < H <= limitMax and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl
# x就是宽  y就是高 l是起始

# 计算轮廓 外接矩形
def img_findContours(img_contours):
    """
    cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
    返回的list中每个元素都是图像中的一个轮廓
    传进来的图形 .copy一下 方便以后使用
            :param img_contours: 形态学后的图片
            :return: 第一步定位的图片包括（中心(x,y), (宽,高), 旋转角度）
    """

    contours, hierarchy = cv2.findContours(img_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 查找检测题轮廓 轮廓检测方法：建立一个等级树结构 轮廓近似的方法：压缩垂直、水平、对角方向，只保留端点
    cv2.drawContours(img_contours, contours,-1, (255, 182, 193), 3)
    # cv_show("画轮廓",img_contours)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
    # 直接筛选出来面积大于最小车牌面积的区域 当大于最小面积就加入列表
    print("findContours len = ", len(contours))
    car_contours = []
    for cnt in contours:
        ant = cv2.minAreaRect(cnt)
        # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        width, height = ant[1]

        if width < height:
            width, height = height, width
        ration = width / height
        if 2 < ration < 5.5:
            car_contours.append(ant)
    return car_contours
# 这个操作就是为了找到符合条件的车牌位置

#进行矩形矫正 仿射变换
def img_Transform(car_contours, oldimg, pic_width, pic_hight):
    """
               :param car_contours: 外边界轮廓 （中心(x,y), (宽,高), 旋转角度）
               :param oldimg:高斯滤波图
               :param pic_width:图片宽
               :param pic_hight:图片高
               :return: car_imgs返回处理后图片
       """
    car_imgs = []
    for contour in car_contours:
        if -1 < contour[2] < 1:
            angle = 1

        else:
            angle = contour[2]
        contour = (contour[0], (contour[1][0] + 5, contour[1][1] + 5), angle)
        box = cv2.boxPoints(contour)
        # box旋转矩形的 4个顶点（用于绘制旋转矩形的辅助函数）
        heigth_point = right_point = [0, 0]
        # 右上角
        left_point = low_point = [pic_width, pic_hight]
        # 左下角
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point
        # 正角度 height 左下角 right 右下角 left 左上角  low 右上
        # 负角度 height 右下角 right 右上  left 左下 low 左上
        if left_point[1] <= right_point[1]:  # 正角度
            new_right_point = [right_point[0], heigth_point[1]]
            # 右下角
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            # pts1输入的三个点 pts2输出的三个点
            M = cv2.getAffineTransform(pts1, pts2)
            # 仿射矩阵M opencv提供了根据变换前后三个点的对应关系来自动求解M
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            # 对源图像应用上面求得的仿射变换

            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            # 判断是否小于零
            car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            # 第一个参数 设置目标图像的大小和类型与源图像一致  carimg就是车牌区域
            # cv_show('img',car_img)
            car_imgs.append(car_img)

        elif left_point[1] > right_point[1]:  # 负角度
            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            car_imgs.append(car_img)

    return car_imgs

# 车牌颜色判别
def img_color(card_imgs):
    """
        :param card_imgs: 外边界轮廓
        :return: colors  颜色
        :return:card_imgs 颜色区域图片
    """
    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        # 有转换失败的可能，原因来自于上面矫正矩形出错
        if card_img_hsv is None:
            continue
        row_num, col_num = card_img_hsv.shape[:2]
        card_img_count = row_num * col_num

        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 11 < H <= 34 and S > 34:
                    yello += 1
                elif 35 < H <= 99 and S > 34:
                    green += 1
                elif 99 < H <= 124 and S > 34:
                    blue += 1

                if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"
        limitMin = limitMax = 0
        if yello * 2 >= card_img_count:
            color = "yello"
            limitMin = 11
            limitMax = 34
        elif green * 2 >= card_img_count:
            color = "green"
            limitMin = 35
            limitMax = 99
        elif blue * 2 >= card_img_count:
            color = "blue"
            limitMin = 100
            limitMax = 124
        elif black + white >= card_img_count * 0.7:
            color = "by"
        colors.append(color)
        card_imgs[card_index] = card_img
        #判断车牌颜色

        if limitMin == 0:
            continue
        xl, xr, yh, yl = accurate_place(card_img_hsv, limitMin, limitMax, color)
        # yl xl 左上角起点  xr宽 yh高
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        # 是否需要定位
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        if color == "green":
            card_imgs[card_index] = card_img
        else:
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh,xl:xr]

        if need_accurate:
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limitMin, limitMax, color)
            # print("11xl", xl, "xr", xr, "yh", yh, "yl", yl)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        if color == "green":
            card_imgs[card_index] = card_img
        else:
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh,xl:xr]

    return colors, card_imgs


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    """
        :param threshold:阈值
        :param histogram: 柱状图高
        :return:wave_peaks：返回波峰的各个范围
    """
    up_point = -1  # 上升点
    is_peak = False
    # 是否为波峰
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for index, element in enumerate(histogram):
        if is_peak and element < threshold:
            if index - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, index))
        elif not is_peak and element >= threshold:
            is_peak = True
            up_point = index
    if is_peak and up_point != -1 and index - up_point > 4:
        wave_peaks.append((up_point, index))
    return wave_peaks

#分离车牌字符
def seperate_card(img, waves):
    """
            :param img:要分割的图片
            :param waves: 波峰图
            :return:part_cards：分割的字符
    """
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards



