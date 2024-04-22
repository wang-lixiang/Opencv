import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
# 现实中人两眼中心的距离是6.3cm
W = 6.3

while True:
    success, img = cap.read()
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

    cv2.imshow('Image', img)
    cv2.waitKey(1)
