import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('TargetImage.jpg')
myVid = cv2.VideoCapture('2.mp4')

detection = False
frameCounter = 0

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

# 创建一个用于关键点检测器对象，将最大特征数量设置为1000
orb = cv2.ORB_create(nfeatures=1000)
# kp1将是表示检测到的关键点的KeyPoint对象列表。
# des2将是一个包含每个关键点的计算描述符的NumPy数组。
# None作为掩码参数传递，表示在检测期间不使用掩码。
kp1, des1 = orb.detectAndCompute(imgTarget, None)


# cv2.drawKeypoints函数将在图像上绘制检测到的关键点，以便于可视化
# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)

# 将图像数组堆叠成一个大图像，并添加标签
def stackImages(imgArray, scale, lables=[]):
    # 获取图像数组中的第一个图像的宽度与高度
    sizeW = imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    # 检查图像数组中的第一个元素是否是列表，以确定图像数组是否为二维数组
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 255), 2)
    return ver


while True:
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)

    if detection == False:
        # 用于将视频的帧位置设置为指定的位置，这里是第一帧的位置
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        # 检查帧计数器是否等于视频的总帧数
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    # 创建一个Brute Force匹配器
    bf = cv2.BFMatcher()
    # 这里的k=2表示返回两个最佳匹配点，即对于每个查询描述符，返回两个最近邻的训练描述符
    matches = bf.knnMatch(des1, des2, k=2)

    # 存储最佳匹配点
    good = []
    # 对于每个匹配对，检查第一个匹配点的距离是否小于第二个匹配点的距离的0.75倍
    # 如果是，则将该匹配点添加到good列表中
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    # 使用cv2.drawMatches函数绘制特征匹配结果
    # imgTarget是第一幅图像，kp1是该图像的关键点，imgWebcam是第二幅图像，kp2是该图像的关键点
    # good是匹配列表，None表示不使用掩码，flags=2表示绘制匹配点和关键点
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

    # 检查是否有足够的好的匹配点数
    if len(good) > 20:
        detection = True
        # 获取好的匹配点的源图像和目标图像坐标
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 使用RANSAC算法估计两个图像之间的单应性矩阵（描述两个平面之间的投影关系）
        # 参数cv2.RANSAC表示使用RANSAC算法，5表示最大迭代次数
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        # pts 是目标图像的四个角点的坐标
        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        # 将目标图像的四个角点根据给定的单应性矩阵 matrix 进行透视变换，以获取在源图像中的对应坐标
        dst = cv2.perspectiveTransform(pts, matrix)
        # 绘制了源图像中目标图像的轮廓，使用了变换后的坐标，线条颜色为紫色，线条宽度为 3 像素，True 表示绘制闭合的多边形
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)
        cv2.imshow('img2', img2)

        # 将输入图像 imgVideo 根据给定的单应性矩阵 matrix 进行透视变换
        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        # 创建纯黑模板
        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        # 将我们摄像头识别到的指定区域填充为白色
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        # 取反，指定区域变黑，其他区域变白
        maskInv = cv2.bitwise_not(maskNew)
        # 使用掩码将源图像中目标区域之外的部分设为黑色
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        # 将透视变换后的目标图像与源图像中的目标区域进行合成
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

        imgStacked = stackImages(([imgWebcam, imgVideo, imgTarget], [imgFeatures, imgWarp, imgAug]), 0.5)

    cv2.imshow('imgStacked', imgStacked)
    cv2.waitKey(1)
    frameCounter += 1
