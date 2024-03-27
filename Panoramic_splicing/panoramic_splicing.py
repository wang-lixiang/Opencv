import cv2
import os

mainFolder = 'Images'
# 读取到images文件下的子文件，每个文件夹表示需要拼接到一起的图片集合
myFolders = os.listdir(mainFolder)

for folder in myFolders:
    path = mainFolder + '/' + folder
    images = []
    myList = os.listdir(path)
    # f是一种字符串格式化方式，避免了字符串拼接的乱象
    print(f'Total no of images detected {len(myList)}')
    for imgN in myList:
        curImg = cv2.imread(f'{path}/{imgN}')
        # (0, 0) 表示输出图像的尺寸与输入图像的尺寸成比例，即按照指定的缩放因子进行缩放
        # None：表示不指定输出图像的尺寸类型，即输出图像的尺寸由 (0, 0) 和缩放因子确定
        # 0.2, 0.2：表示水平和垂直方向的缩放因子，这里分别为 0.2，即将图像缩放为原始大小的 20%
        curImg = cv2.resize(curImg, (0, 0), None, 0.2, 0.2)
        images.append(curImg)
    # 创建了一个 Stitcher 对象。这个对象用于执行图像拼接的操作
    stitcher = cv2.Stitcher.create()
    # 使用 Stitcher 对象的 stitch 方法来拼接图像
    (status, result) = stitcher.stitch(images)
    # status：拼接操作的状态,有两种：cv2.Stitcher_OK：拼接成功；cv2.Stitcher_ERR_NEED_MORE_IMGS：需要更多的图像来完成拼接
    if (status == cv2.STITCHER_OK):
        print('Panorama Generated')
        # 如果拼接成功，则 result 是一个包含全景图像的 Mat 对象；如果拼接失败或需要更多图像，则 result 为 None
        cv2.imshow(folder, result)
        cv2.waitKey(1)
    else:
        print('Panorama Generation Unsuccessful')
cv2.waitKey(0)
