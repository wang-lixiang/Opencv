import cv2
import time
import numpy as np
from comtypes import CLSCTX_ALL

import handtracking_module as htm
import math
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

# 获取系统中的扬声器设备
devices = AudioUtilities.GetSpeakers()
# 激活扬声器设备的音频端点对象
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# 将激活的音频端点对象转换为 IAudioEndpointVolume 接口的指针
volume = cast(interface, POINTER(IAudioEndpointVolume))
# 获取系统音量的范围，返回一个元组，其中包含最小音量和最大音量
volRange = volume.GetVolumeRange()
# 最小音量和最大音量
minVol = volRange[0]
maxVol = volRange[1]

# 映射后的音量
vol = 0
# 映射后的音量条高度，400表示0音量
volBar = 400
# 映射后的音量百分比
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # 进行手势控制的两个指尖定位并标记
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 计算手指间的直接的距离
        length = math.hypot(x2 - x1, y2 - y1)

        # 手指之间的距离范围是 50 - 300
        # 音量的范围是 -63.5 - 0

        # np.interp() 函数用于线性插值
        # [50, 300]是length的范围，映射到指定的范围上，返回值是映射后的值
        # 映射后的音量值
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        # 映射后的音量条高度
        volBar = np.interp(length, [50, 300], [400, 150])
        # 映射后的音量百分比
        volPer = np.interp(length, [50, 300], [0, 100])

        # 将得到的音量值设置为系统的主音量值
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            # 如果手指之间距离比较短，标记为红色，BGR
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

    # 音量条的波动
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # 显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
