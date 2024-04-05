import cv2
import numpy as np


# 经过一系列变换，获得图像中满足条件的轮廓
def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使图像变得平滑/降噪，卷积核大小为（5，5），1表示标准差
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # 使用Canny边缘检测检测边缘
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    # 创建膨胀操作的内核
    kernel = np.ones((5, 5))
    # 对Canny检测到的边缘进行膨胀，膨胀了3次
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    # 对膨胀后的图像进行腐蚀操作以增强轮廓，腐蚀了2次
    imgThre = cv2.erode(imgDial, kernel, iterations=2)

    # 如果要显示Canny边缘图像
    if showCanny:
        cv2.imshow('Canny', imgThre)

    # 在阈值化后的图像种查找轮廓,contours为轮廓信息，hierarchy为层级
    # cv2.RETR_EXTERNAL: 指定轮廓的提取模式。在这个模式下，函数只会返回最外层（外部）的轮廓，不会返回任何孔（内部）的轮廓
    # cv2.CHAIN_APPROX_SIMPLE: 这是指定轮廓的近似方法。函数会返回一个由轮廓上的顶点组成的列表，其中的每个顶点就是轮廓上的一个端点。
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finalCountours = []
    # 遍历找到的每个轮廓
    for i in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(i)

        # 检查面积是否大于指定的最小面积
        if area > minArea:
            # 将轮廓近似为一个简单的多边形
            # i 是指定的轮廓， True 表示该轮廓是一个封闭的曲线，计算该轮廓的周长
            peri = cv2.arcLength(i, True)
            # 用较少的点来近似表示原始轮廓，0.02 * peri 是指定逼近精度，True表示封闭
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)

            # 将轮廓包围在最小的矩形内部,返回一个矩形，表示轮廓的边界框,矩形的左上角顶点的坐标（x，y），以及矩形的宽度和高度
            bbox = cv2.boundingRect(approx)

            # 检查轮廓中的顶点数量是否与指定的过滤器匹配
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])

    # 根据面积的降序对最终轮廓进行排序
    # 指定了一个匿名函数，该函数将列表中每个元素 x 的第二个值作为比较的依据
    # reverse=True 参数指定了按照降序排序，也就是面积最大的轮廓排在最前面
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)

    if draw:
        for con in finalCountours:
            # con[4]: 这是要绘制的轮廓，是一个由轮廓上的点组成的数组
            # -1: 这个参数表示绘制所有轮廓。如果设定为正数，则表示只绘制具有特定索引的轮廓。
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalCountours


# 重新排序四个顶点，使其顺序为左上、右上、左下、右下
def reorder(myPoints):
    # 接受一个数组作为参数，并返回一个与该数组具有相同形状和数据类型的全零数组
    myPointsNew = np.zeros_like(myPoints)
    # 将输入的四个顶点坐标重塑为一个4*2的矩阵
    myPoints = myPoints.reshape((4, 2))
    # 返回一个包含四个值的数组，分别是每行中两个元素的和，1表示的是按行求和，4*1
    add = myPoints.sum(1)
    # 找到和最小的顶点，即左上角顶点，放到新顺序的第一个位置
    myPointsNew[0] = myPoints[np.argmin(add)]
    # 找到和最大的顶点，即右下角顶点，放到新顺序的第四个位置
    myPointsNew[3] = myPoints[np.argmax(add)]
    # 返回一个形状为 (4, 1) 的数组，其中每个元素是相邻两个元素之间的差异
    diff = np.diff(myPoints, axis=1)
    # 找到差值最小的顶点，即左下角顶点，放到新顺序的第二个位置
    myPointsNew[1] = myPoints[np.argmin(diff)]
    # 找到差值最大的顶点，即右上角顶点，放到新顺序的第三个位置
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


# 返回透视变换并裁剪后的图像
def warpImg(img, points, w, h, pad=20):
    # 重新排序四个顶点，使其顺序为左上、右上、左下、右下
    points = reorder(points)
    # 将顶点坐标转换为浮点数类型
    pts1 = np.float32(points)
    # 设置图像的四个顶点坐标，按照左上、右上、左下、右下的顺序
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # 分别是原始图像中的四个顶点和目标图像中的对应四个顶点
    # 返回的 matrix 是一个 3x3 的变换矩阵，描述了如何将原始图像中的点映射到目标图像中
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 目标图像的尺寸 (w, h)。返回的 imgWarp 是经过透视变换后的图像
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    # 通过切片操作去除了左右各 pad 个像素的边界填充部分,高度和宽度
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    return imgWarp


# 计算两点之间的欧式距离
def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5
