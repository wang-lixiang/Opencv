import math
import random

import cv2
import cvzone
from HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathfood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point

        # 当处理包含透明通道,为了保证读取的图像数据完整性和准确性时
        # cv2.IMREAD_UNCHANGED指定了读取图像时应该保持原始的通道数和数据类型
        self.imgFood = cv2.imread(pathfood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        # 初始的分数
        self.score = 0
        self.gameOver = False

    # 随机食物的位置
    def randomFoodLocation(self):
        # 用于生成指定范围内的随机整数
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):

        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [300, 400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Your Score:{self.score}', [300, 550], scale=7, thickness=5, offset=20)
        else:

            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            # hypot()函数返回两个参数的平方和的平方根
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            # 更新长度
            self.currentLength += distance
            # 更新上一个结点
            self.previousHead = cx, cy

            # 控制长度为限定值allowedLength
            if self.currentLength > self.allowedLength:
                # 逐渐消除最早的点来控制长度
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    # 从列表 self.lengths 中移除索引为 i 的元素
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            # 检查蛇是否吃了食物
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                # 更新食物的下一个点
                self.randomFoodLocation()
                # 蛇的允许长度增加
                self.allowedLength += 50
                # 分数增加
                self.score += 1
                print(self.score)

            # 画出这条蛇
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                cv2.circle(imgMain, self.points[-1], 20, (200, 0, 200), cv2.FILLED)

            # 画出食物
            # 将 self.imgFood 叠加到 imgMain 上,图像叠加，后面指定了位置
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))
            cvzone.putTextRect(imgMain, f'Score:{self.score}', [50, 80], scale=3, thickness=3, offset=10)

            # 检查是否蛇头和蛇身发生碰撞
            # 从self.points中获取前面除了最后两个元素之外的所有点，并将它们转换为NumPy数组
            pts = np.array(self.points[:-2], np.int32)
            # 每个点都是一个包含两个坐标的列表
            pts = pts.reshape((-1, 1, 2))
            # 利用上面的pts绘制多边形，False表示不闭合
            cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
            # 用于计算点 (cx, cy) 到多边形 pts 的最短距离
            # 参数 True 表示计算的是有符号的距离，即距离在多边形内部的点为正值，距离在多边形外部的点为负值，距离在多边形上的点为零
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -1 <= minDist <= 1:
                print("hit")
                self.gameOver = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed Length
                self.previousHead = 0, 0  # previous head point
                self.randomFoodLocation()

        return imgMain


game = SnakeGameClass("Donut.png")

while True:
    success, img = cap.read()
    # 将图像img水平翻转。参数1表示水平翻转，如果将参数改为0，则是垂直翻转
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        # 获取第一只手的信息
        lmList = hands[0]['lmList']
        # 存储食指的指尖的x,y
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    # 按下r键即可重来
    if key == ord('r'):
        game.gameOver = False
        game.score = 0
