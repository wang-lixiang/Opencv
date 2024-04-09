# *表示所有其他函数、类、变量等都将被导入到当前的命名空间中，可以直接使用而不需要通过模块名限定
from utlis import *
import pytesseract

path = '1.png'
hsv = [0, 65, 59, 255, 0, 255]
pytesseract.pytesseract.tesseract_cmd = 'E:\\Tesseract-OCR\\tesseract.exe'


img = cv2.imread(path)
# 在给定的图像中检测出指定HSV颜色范围内的物体，并返回一个只包含这些颜色的图像
imgResult = detectColor(img, hsv)
imgContours, contours = getContours(imgResult, img, showCanny=True,
                                    minArea=1000, filter=4,
                                    cThr=[100, 150], draw=True)
cv2.imshow("imgContours", imgContours)
print(len(contours))

#### Step 5 ####
roiList = getRoi(img, contours)
# cv2.imshow("TestCrop",roiList[2])
roiDisplay(roiList)

#### Step 6 ####
highlightedText = []
for x, roi in enumerate(roiList):
    # print(pytesseract.image_to_string(roi))
    highlightedText.append(pytesseract.image_to_string(roi))

saveText(highlightedText)

imgStack = stackImages(0.7, ([img, imgResult, imgContours]))
cv2.imshow("Stacked Images", imgStack)

cv2.waitKey(0)
