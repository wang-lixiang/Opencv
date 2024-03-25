import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
# 设置视频流的宽度为 640 像素
cap.set(3, 640)
# 设置视频流的高度为 480 像素
cap.set(4, 480)

while True:
    success, img = cap.read()
    for barcode in decode(img):
        # barcode.data得到的数据是二进制的，转化为utf-8形式显示
        myData = barcode.data.decode('utf-8')
        print(myData)
        # 从 barcode 对象中获取多边形的顶点坐标，并将其存储在 NumPy 数组中
        pts = np.array([barcode.polygon], np.int32)
        # 将其变为三维数组。这是因为 cv2.polylines() 函数要求传入的多边形坐标是三维数组的形式
        pts = pts.reshape((-1, 1, 2))
        # 绘制所有的闭合(True)多边形
        cv2.polylines(img, [pts], True, (255, 0, 255), 5)
        # 获取矩形坐标，不使用多边形是因为不能让文字也旋转
        pts2 = barcode.rect
        # 在指定位置写下扫描出来的文字
        cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow('Result', img)
    cv2.waitKey(1)
