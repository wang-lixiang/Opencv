import cv2
import pytesseract

# 获取本地的Tesseract-OCR文件
pytesseract.pytesseract.tesseract_cmd = 'E:\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 识别图像上的信息并打印出来
# print(pytesseract.image_to_string(img))
# 识别图像上每个字符，并返回字符，左下角坐标XY，宽度，长度，置信度
# print(pytesseract.image_to_boxes(img))

# Detecting Characters
hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    # 循环取每一个字符的信息，再按空格划分为数组
    b = b.split(' ')
    # 取字符的左下角坐标，宽高
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    print(x, y, w, h)
    # 画出矩形框
    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)
    # 在每个框下显示对应的字符
    cv2.putText(img, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

# # Detecting Words
# hImg, wImg, _ = img.shape
# # --oem 3: 这个参数指定了 OCR 引擎的模式。在这里，3 表示使用 Tesseract 的默认 LSTM OCR 引擎
# # --psm 6: 这个参数指定了 OCR 的识别模式。在这里，6 表示按行识别文本块
# # outputbase digits: 这个参数指定了 OCR 的输出基础设置，将 OCR 结果仅限制为数字
# cong = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_data(img, config=cong)
# for x, b in enumerate(boxes.splitlines()):
#     if x != 0:
#         # 循环取每一个字符的信息，再按空格划分为数组
#         b = b.split()
#         if len(b) == 12:
#             # 取字符的左下角坐标，宽高
#             x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#             # 画出矩形框
#             cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)
#             # 在每个框下显示对应的字符
#             cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
#
cv2.imshow('Result', img)
cv2.waitKey(0)
