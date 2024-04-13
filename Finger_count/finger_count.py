import cv2
import handtracking_module as htm
import os
import time

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "finger_images"
# 读取文件夹中图片的名字并形成一个列表
myList = os.listdir(folderPath)

# 将所有图片存放为一个列表
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime = 0
# 实例化一个检测手指的对象
detector = htm.handDetector(detectionCon=0.75)

# 对应每个手指的指尖序号
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # 大拇指闭合张开判断，若指尖y坐标比指节y坐标高，则为张开 1
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 其它四指张开闭合判断，依靠z轴坐标进行判断
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 统计手指状态为1的个数
        totalFingers = fingers.count(1)

        # 根据手指数量在视频帧上叠加相应的手指图像
        h, w, _ = overlayList[totalFingers - 1].shape
        # 图像指定区域进行图像替换
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        # 在视频上绘制手指数量文本
        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
