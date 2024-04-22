import cv2
import cvzone
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
# 现实中人两眼中心的距离是6.3cm
W = 6.3

textList = ["Hello", "world!"]

# 设置文字缩放的敏感度，避免抖动，离散型改变大小
sen = 25

while True:
    success, img = cap.read()
    # 创建与原始图像相同大小的黑色图像，用于绘制文字
    imgtext = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        # 获取第一张人脸的脸部坐标
        face = faces[0]
        # 获取左右眼的中心坐标
        LeftEyepoints = face[145]
        RightEyepoints = face[347]

        # 对于camera获取的图片中，两眼之间的距离是w
        w, _ = detector.findDistance(LeftEyepoints, RightEyepoints)
        # 可以设定一下d=50,计算出大致的焦距f = (w * d) / W
        f = 840
        # 反过来计算实际眼部距离屏幕的距离
        d = (W * f) / w

        # 实时显示距离
        cvzone.putTextRect(img, f'distance:{int(d)}cm', (face[10][0] - 100, face[10][1] - 60), scale=2)

        for i, text in enumerate(textList):
            singleHeight = 20 + int((int(d / sen) * sen) / 4)
            scale = 0.4 + (int(d / sen) * sen) / 75
            cv2.putText(imgtext, text, (50, 50 + i * singleHeight), cv2.FONT_ITALIC, scale, (255, 255, 255), 2)

    imgStack = cvzone.stackImages([img, imgtext], 2, 1)
    cv2.imshow('Image', imgStack)
    cv2.waitKey(1)
