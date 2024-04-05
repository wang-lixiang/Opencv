import cv2
import utlis

webcam = False
# 这里存放了一个样例的操作方式，你可以修改webcam为False来查看
path = '1.png'
# 当你外接一个摄像头的时候，可以修改为1以此获得外接摄像头的使用
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)
# 设定透视变换后的图像大小,也就是A4的标准像素
scale = 3
wP = 210 * scale
hP = 297 * scale

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    # 获取图像中的A4纸张轮廓
    imgContours, conts = utlis.getContours(img, minArea=50000, filter=4)
    if len(conts) != 0:
        # 获取最大的轮廓（即A4纸张轮廓）
        biggest = conts[0][2]
        # 将A4纸张进行透视变换，变换为标准大小
        imgWarp = utlis.warpImg(img, biggest, wP, hP)

        # 在透视变换后的图像中检测物体轮廓
        imgContours2, conts2 = utlis.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)
        if len(conts2) != 0:
            for obj in conts2:
                # 在图像上绘制多边形，[obj[2]]：这是一个包含多边形顶点坐标的列表,  True 表示要绘制闭合的多边形
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                # 重新排序多边形顶点，4*1*2
                nPoints = utlis.reorder(obj[2])
                # 计算物体的尺寸，换算成cm
                nW = round((utlis.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
                nH = round((utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)

                # 绘制尺寸信息箭头线,8：线段的终点样式，这里是箭头终点,0：线段的偏移量，表示箭头长度和线段长度之间的比例关系
                # 0.05：线段的箭头比例，表示箭头长度与箭头宽度的比值
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]), (255, 0, 255), 3, 8, 0, 0.05)

                # 绘制尺寸信息文本
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        # 显示处理后的图像
        cv2.imshow('A4', imgContours2)

    # 将图像缩小原来的一半，即1/4
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow('Original', img)
    cv2.waitKey(1)
