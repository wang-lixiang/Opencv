import cv2
import math

path = "test.png"
img = cv2.imread(path)
pointlist = []


# 点击时需先点角点，在点另外两个点
# 鼠标点击获得的点
def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointlist)
        if size != 0 and size % 3 != 0:
            # 元组是用来确定起始坐标，也就是画两条线时始终以角点为起始点
            cv2.line(img, tuple(pointlist[round((size - 1) / 3) * 3]), (x, y), (0, 0, 255), 2)
        # cv2.FILLED: 指定填充整个圆形
        cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
        pointlist.append([x, y])


# 求两点连成的线的斜率
def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


# 获取角度
def getAngle(pointlist):
    pt1, pt2, pt3 = pointlist[-3:]
    # 计算组成角度的两条直线的斜率
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    # 利用斜率计算出角度
    if (m2 - m1) != 0:
        angR = math.atan(abs((m2 - m1) / (1 + (m2 * m1))))
        angD = round(math.degrees(angR))
    else:
        # 发现是直角，tan90是不能计算的
        angD = 90

    cv2.putText(img, str(angD), (pt1[0] - 40, pt1[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)


while True:
    if len(pointlist) % 3 == 0 and len(pointlist) != 0:
        getAngle(pointlist)
    cv2.imshow('Image', img)
    # 可以实现在图像上捕获鼠标事件，比如点击、拖动、释放等，并在回调函数中执行相应的操作
    cv2.setMouseCallback('Image', mousePoints)
    # 当 delay > 0 时，表示等待指定的毫秒数后继续执行程序。
    # 当 delay = 0 时，表示无限期等待键盘按键事件，直到用户按下一个键。
    # 当 delay < 0 时，表示不会等待键盘按键事件，并立即继续执行程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointlist = []
        img = cv2.imread(path)
