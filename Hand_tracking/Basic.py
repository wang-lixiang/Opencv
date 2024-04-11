import cv2
# 谷歌的一个库，用于各种感知任务
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# 人手总的有20个坐标，可以参考img.png
mpHands = mp.solutions.hands
# 实例化一个手部追踪
hands = mpHands.Hands()
# 被赋予绘图工具模块
mpDraw = mp.solutions.drawing_utils

# 上一帧时间和当前帧的时间，用于计算每秒的帧数
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    # Mediapipe 库处理的图像格式是 RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 识别图像中的手，并将结果保存至results
    results = hands.process(imgRGB)

    # 是否存在多个手部的标志点。如果有，说明在图像中检测到了手部
    if results.multi_hand_landmarks:
        # 通过迭代对每一个手部的标志点进行处理
        for handLms in results.multi_hand_landmarks:
            # 枚举遍历每个手部标志点，并为每个标志点分配一个唯一的 id，以及其相应的位置坐标
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                # lm.x 和 lm.y 是相对于图像宽度和高度的归一化坐标, 相乘之后变成图像中实际的坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 着重显示大拇指点
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            # 要绘制的图像，手部标志点的列表，标志点之间连接关系
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # 计算帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # 显示帧率
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('img',img)
    cv2.waitKey(1)
