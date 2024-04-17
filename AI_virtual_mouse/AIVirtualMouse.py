import cv2
import numpy as np
import handtracking_module as htm
import time
import autopy

# 获取当前屏幕的尺寸
wScr, hScr = autopy.screen.size()
# 显示边框的大小
wCam, hCam = 640, 480
# 基础距离，距离显示边框的距离
frameR = 100

# 平滑系数 smoothening 来减小鼠标移动的速度，以使移动更加平滑
smoothening = 7
pTime = 0
# plocX plocY 是上一帧的鼠标横纵坐标
plocX, plocY = 0, 0
# clocX clocY 当前帧的鼠标横纵坐标
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# 实例化一个手部检测器，最大检测手部为1
detector = htm.handDetector(maxHands=1)

while True:
    success, img = cap.read()
    # 识别到手关节并画出来
    img = detector.findHands(img)
    # 定位手部21个位置的空间坐标z,x,y
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 获取手指此刻状态，伸展1还是弯曲0，返回的是一个01数组
        fingers = detector.fingersUp()
        # 手指可操控范围对应整个屏幕
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 食指伸直，中指弯曲时可以移动鼠标
        if fingers[1] == 1 and fingers[2] == 0:
            # 将x1从在(frameR, wCam - frameR)范围内的值映射到(0, wScr)范围内的值
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 计算平滑移动后的鼠标横纵坐标
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 移动鼠标到指定位置，水平位置被设置为 wScr - clocX，垂直位置为 clocY
            autopy.mouse.move(wScr - clocX, clocY)
            # 绘制食指指尖位置
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            # 更新上一次鼠标位置
            plocX, plocY = clocX, clocY

        # 食指和中指同时伸直
        if fingers[1] == 1 and fingers[2] == 1:
            # 计算两个指尖指尖的距离 lineInfo是一个数组[x1, y1, x2, y2, cx, cy]
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 当两点之间距离大于指定值时鼠标点击
            if length < 20:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    # 显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
