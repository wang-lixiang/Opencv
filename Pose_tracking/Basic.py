import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
# 追踪人体的坐标
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# 这里接入摄像头，可以使用test.mp4测试
cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # 只能识别一个人体并显示
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
