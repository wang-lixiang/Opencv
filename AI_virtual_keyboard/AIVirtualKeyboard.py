import cv2
from HandTrackingModule import HandDetector
from time import sleep
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

# 虚拟键盘上获取的字符串
finalText = ""

# 初始化键盘控制器
keyboard = Controller()


# 画出所有按钮
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


class Button():
    def __init__(self, pos, text, size=[85, 85]):
        # 按钮左上角位置
        self.pos = pos
        # 按钮大小
        self.size = size
        # 按钮上的文本
        self.text = text


# 创建按钮文本
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)
    img = drawAll(img, buttonList)

    # lmList存储的是手部每个点的x,y坐标
    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            # 检查食指指尖是否在按钮区域内
            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                # 计算食指和中指指尖之间的距离
                l, _, _ = detector.findDistance(8, 12, img, draw=False)

                if l < 40:
                    # 模拟实际键盘按键的效果，也就可以用来输入文本达到键盘的效果
                    keyboard.press(button.text)

                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    finalText += button.text
                    sleep(0.3)

    # 在控制屏上显示按下的字符串
    # cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    # cv2.putText(img, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
