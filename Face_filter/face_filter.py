import cv2
import numpy as np
import dlib

webcam = True
cap = cv2.VideoCapture(0)

# 基于一个经过训练的分类器，可以快速准确地检测图像中是否存在人脸，并返回检测到的人脸位置
detector = dlib.get_frontal_face_detector()
# 加载了一个面部特征点预测器模型, 包含了用于检测人脸68个关键特征点的参数和信息。这些特征点包括眼睛、眉毛、鼻子、嘴巴等部位的位置
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def empty(a):
    pass


# 创建了一个名为 "BGR" 的窗口，用于显示图像, 这个窗口将用于显示我们后续处理的图像
cv2.namedWindow("BGR")
# 调整窗口的大小为640*480
cv2.resizeWindow("BGR", 640, 480)
# 创建了一个名为 "Blue" 的滑动条，将其附加到 "BGR" 窗口上，初始值为153，最大值为255
cv2.createTrackbar("Blue", "BGR", 153, 255, empty)
# 当滑动条的值发生变化时，会调用 empty 函数
cv2.createTrackbar("Green", "BGR", 0, 255, empty)
cv2.createTrackbar("Red", "BGR", 137, 255, empty)


def createBox(img, points, scale=5, masked=False, cropped=True):
    if masked:
        # 创建一个原始图像具有相同大小和通道数的全黑掩码
        mask = np.zeros_like(img)
        # [points] 是多边形的顶点坐标列表，这些顶点定义了要填充的区域，(255, 255, 255)即白色
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        # 特定区域为白色，合二为一
        img = cv2.bitwise_and(img, mask)
    if cropped:
        # 计算包围多边形区域的最小矩形边界框。这个函数返回边界框的左上角坐标 (x, y) 和宽度 w、高度 h
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        # 从原始图像中提取特定的区域
        imgCrop = img[y:y + h, x:x + w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        cv2.imwrite("Mask.jpg", imgCrop)
        return imgCrop
    else:
        return mask


while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread('1.png')
    img = cv2.resize(img, (0, 0), None, 0.6, 0.6)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 识别图片中的每一个人脸,是一个列表
    faces = detector(imgOriginal)
    # 对每一张人脸进行判断
    for face in faces:
        # 用预先训练好的人脸关键点检测器对灰度图像中的人脸进行关键点检测，face作为一个矩形区域指定在哪个区域检测
        landmarks = predictor(imgGray, face)
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])
        if len(myPoints) != 0:
            try:
                myPoints = np.array(myPoints)
                # # 提取左眉毛区域并展示
                # imgEyeBrowLeft = createBox(img, myPoints[17:22])
                # cv2.imshow('Left Eyebrow', imgEyeBrowLeft)

                # # 其它的面部区域
                # imgEyeBrowRight = createBox(img, myPoints[22:27])
                # imgNose = createBox(img, myPoints[27:36])
                # imgLeftEye = createBox(img, myPoints[36:42])
                # imgRightEye = createBox(img, myPoints[42:48])
                # imgLips = createBox(img, myPoints[48:61])
                # cv2.imshow('Right Eyebrow', imgEyeBrowRight)
                # cv2.imshow('Nose', imgNose)
                # cv2.imshow('Left Eye', imgLeftEye)
                # cv2.imshow('Right Eye', imgRightEye)
                # cv2.imshow('Lips', imgLips)

                maskLips = createBox(img, myPoints[48:61], masked=True, cropped=False)
                # 创建一个与嘴唇掩码大小相同的全黑图像
                imgColorLips = np.zeros_like(maskLips)
                # 获取滑动条 "Blue"、"Green" 和 "Red" 的当前值，即用户选择的颜色
                b = cv2.getTrackbarPos("Blue", "BGR")
                g = cv2.getTrackbarPos("Green", "BGR")
                r = cv2.getTrackbarPos("Red", "BGR")

                # 将嘴唇颜色图像的所有像素值设置为用户选择的颜色
                imgColorLips[:] = b, g, r
                # 将嘴唇颜色图像与嘴唇掩码进行按位与操作，以保留嘴唇区域的颜色
                imgColorLips = cv2.bitwise_and(maskLips, imgColorLips)
                # 对填充颜色后的嘴唇图像进行高斯模糊处理，使得填充的颜色更加自然
                imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)

                # 确保各个图像具有相同的通道数和颜色空间，以便进行正确的图像处理操作
                imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
                imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
                #  1，表示完全保留原始图像, 0.4，表示对第二个输入图像的叠加透明度; 0是常数，表示输出图像中每个像素值都会加上这个常数
                # 这里可以将参数换成imgOriginal->imgOriginalGray，显示灰度图
                imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.4, 0)
                cv2.imshow('BGR', imgColorLips)
            except:
                pass
    # cv2.imshow("Original", imgOriginal)
    cv2.waitKey(1)
