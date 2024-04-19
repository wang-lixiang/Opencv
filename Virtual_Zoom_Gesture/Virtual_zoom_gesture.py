import cv2
from HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
# 用于存储两个手指之间的初始距离
startDist = None
# 用于图像的缩放
scale = 0
# 放置图像的中心坐标
cx, cy = 500, 500

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    img1 = cv2.imread("img.png")

    if len(hands) == 2:
        # 两只手的食指和拇指要保持伸直
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]

            if startDist is None:
                length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                startDist = length
            length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
            # 更新缩放因子
            scale = int((length - startDist) // 2)
            # 更新中心坐标
            cx, cy = info[4:]
            print(scale)
    # 如果只检测到一只手或没有手，则将 startDist 重置为 None
    else:
        startDist = None

    # 图像不存在或者未成功加载等图像的异常需要处理
    try:
        h1, w1, _ = img1.shape
        # 计算新的高度和宽度，以便将图像调整为偶数大小
        newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
        img1 = cv2.resize(img1, (newW, newH))

        # 将调整大小后的图像 img1 放置在 img 的指定区域。这个区域是以 (cx, cy) 为中心，大小为 (newW, newH) 的矩形区域
        img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
    except Exception as e:
        print("Error:", e)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
