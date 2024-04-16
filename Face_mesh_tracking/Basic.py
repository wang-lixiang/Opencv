import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('test.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
# 创建一个绘制规范
drawSpec = mpDraw.DrawingSpec((0,255,0),thickness=1, circle_radius=1)
mpFaceMesh = mp.solutions.face_mesh
# 最多检测两张人脸，并且对每张人脸都进行关键点检测
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        # faceLms是包含了一张人脸的所有的landmark
        for faceLms in results.multi_face_landmarks:
            # 绘制检测到的关键点,faceLms是一个包含检测到的人脸关键点的数据结构, FACEMESH_CONTOURS 表示绘制人脸的轮廓
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            # 输出一张人脸上每个点的坐标
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, _ = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

    # 显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(50)
