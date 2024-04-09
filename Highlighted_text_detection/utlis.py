import cv2
import numpy as np


# 在给定的图像中检测出指定HSV颜色范围内的物体，并返回一个只包含这些颜色的图像
def detectColor(img, hsv):
    # 将图像转换为HSV颜色空间
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 根据给定的HSV阈值范围确定掩码的上下界
    upper = np.array([hsv[1], hsv[3], hsv[5]])
    lower = np.array([hsv[0], hsv[2], hsv[4]])
    # 创建掩码，将HSV图像中指定范围内的颜色设为白色，不在的设为黑色
    mask = cv2.inRange(imgHSV, lower, upper)
    # 将原始图像和掩码进行按位与操作,得到只有在指定HSV范围内的颜色的图像
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    return imgResult


# 返回处理后的绘制图像和筛选后的轮廓列表
def getContours(img, imgDraw, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgDraw = imgDraw.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.array((10, 10))

    # kernel指定膨胀时要考虑的邻域,膨胀的次数是1
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    # cv2.MORPH_CLOSE闭运算，用于关闭图像中的小孔或连接对象,图像中只有黑白两种颜色，并且轮廓以白色显示
    imgClose = cv2.morphologyEx(imgDial, cv2.MORPH_CLOSE, kernel)

    if showCanny:
        cv2.imshow('Canny', imgClose)

    # cv2.RETR_EXTERNAL只检测最外层的轮廓；CHAIN_APPROX_SIMPLE表示对轮廓的边界点进行压缩，只保留重要的轮廓点
    # contours: 这是一个列表，其中每个元素都是一个轮廓的点集；hierarchy: 这是轮廓的层级信息，描述了轮廓之间的关系
    contours, hiearchy = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)

    if draw:
        for con in finalCountours:
            x, y, w, h = con[3]
            cv2.rectangle(imgDraw, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.drawContours(imgDraw, con[4], -1, (0, 0, 255), 2)
    return imgDraw, finalCountours


# 从图像中提取感兴趣区域
def getRoi(img, contours):
    roiList = []
    for con in contours:
        x, y, w, h = con[3]
        roiList.append(img[y:y + h, x:x + w])
    return roiList


# 显示感兴趣的区域
def roiDisplay(roiList):
    for x, roi in enumerate(roiList):
        # 调整ROI的大小为原来的两倍
        roi = cv2.resize(roi, (0, 0), None, 2, 2)
        cv2.imshow(str(x), roi)


# 保存文本到CSV文件中
def saveText(highlightedText):
    with open('HighlightedText.csv', 'w') as f:
        # 遍历每个高亮文本并写入CSV文件
        for text in highlightedText:
            f.writelines(f'\n{text}')


# 将一个图像数组(指定缩放比例)拼接成一个图像并返回
def stackImages(scale, imgArray):
    # 获取图像数组的行数和列数
    rows = len(imgArray)
    cols = len(imgArray[0])
    # 检测图像数组中的每个元素是否是列表，以判断是否有多层图像（每层有多个图像）
    rowsAvailable = isinstance(imgArray[0], list)
    width=imgArray[0][0].shape[1]
    height=imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
