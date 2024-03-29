import cv2
import face_recognition
import os
import datetime

import numpy as np

path = 'ImagesAttendance'
# 将所有的图片存成一个列表
images = []
# 将所有的图片的名字存成一个列表
classNames = []

myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    # os.path.splitext(cl)会将1.jpg划分为1和.jpg两个元素
    # 将所有图片名构成一个列表
    classNames.append(os.path.splitext(cl)[0])


# 图片编码
def findEncodings(images):
    # 将所有的图片的编码存成一个列表
    encodList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 取出第一个识别的人脸的编码，这里img需得RGB格式
        encode = face_recognition.face_encodings(img)[0]
        encodList.append(encode)
    return encodList


# 记录人脸识别的时间记录
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        # f.readlines()读取所有行并返回一个包含所有行内容的列表
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            # 按逗号 , 进行分割，并将分割后的结果存储在名为 entry 的列表中
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            # 返回当前时间的小时、分钟和秒，以 24 小时制的格式表示
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print("Encoding Complete!")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # 用于在图像中定位人脸的位置。它返回一个列表，其中包含每个检测到的人脸的边界框坐标(top, right, bottom, left)
    facesCurFrame = face_recognition.face_locations(imgS)
    # facesCurFrame参数是可以不添加的，未提供，face_recognition 将在获取人脸编码时自动检测图像中的所有人脸
    # 提供的话会检测指定位置的人脸，形成编码
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # 返回一个布尔值的列表，表示每个已知人脸编码是否与要检查的人脸编码匹配
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # 给定一组面部编码，将它们与已知的面部编码进行比较，得到欧氏距离。对于每一个比较的脸，欧氏距离代表了这些脸有多相似
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        # np.argmin()找最小值对应的索引
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            # (top, right, bottom, left)
            y1, x2, y2, x1 = faceLoc
            # 扩大4倍是因为前面缩小了4倍
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # cv2.rectangle左上的坐标和右下的坐标
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    cv2.imshow('Output', img)
    cv2.waitKey(1)
