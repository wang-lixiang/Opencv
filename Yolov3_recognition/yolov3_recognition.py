import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
# 图像大小固定下来为320*320，如果引入其它yolov3，需改变这个全局变量
whT = 320
# 置信度的阈值，只要大于它，我们就认为是对的
confThreshold = 0.5
# 从置信度最大的边界框开始，两个边界框的重叠面积超过了设定的阈值，就保留置信度较大的那个，将低的剔除
nmsThreshold = 0.2
# 种类文件
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    # rstrip('\n') 方法删除末尾的换行符，最后使用 split('\n') 方法将字符串按换行符分割成一个列表
    classNames = f.read().rstrip('\n').split('\n')

# 引入的yolov3-320（也就是输入的图像大小是320*320）的模型参数和构造
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
# 从配置文件和权重文件中加载模型
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# 使用Opencv自带的DNN后端用于执行网络推理的引擎
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# 设置网络首选的推理计算设备为CPU
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# # 如果你的opencv支持cuda，可以使用GPU
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# 在检测到的目标周围绘制边界框并标注类别和置信度
def findObjects(outputs, img):
    hT, wT, _ = img.shape
    # 存储检测到的边界框、类别和置信度的列表
    bbox = []
    classIds = []
    confs = []
    # 分别对3个未连接的输出层进行操作
    for output in outputs:
        # 第一层300个数据，第二层1200，第三层4800
        for det in output:
            # 取出每个类别的得分数
            scores = det[5:]
            # 找到80个类别的最大的类别的索引
            classId = np.argmax(scores)
            # 取出对应的得分数
            confidence = scores[classId]
            # 如果超过了设定的置信度阈值就可以显示
            if confidence > confThreshold:
                # 要乘的原因是输出的值都是百分比
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # 应用非最大抑制（NMS）以过滤重叠的边界框，并返回保留的边界框的索引，返回的是一个二维的列向量n*1
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    # cv.dnn.blobFromImage: 这个函数用于从图像中创建一个4维的blob。Blob是深度学习中一种用于表示图像数据的数据结构
    # 1 / 255: 这是对图像数据进行归一化的因子，将像素值缩放到[0, 1]范围内
    # (whT, whT): 这是目标blob的空间大小，通常与网络模型的输入尺寸相匹配
    # [0, 0, 0]: 这是图像的平均减去的像素值（可选参数），在这里没有进行平均减去操作。
    # 1: 这是图像的缩放因子（可选参数），表示对图像的尺寸进行缩放
    # crop=False: 指定了是否需要对图像进行裁剪以适应目标大小。在这里，设置为False表示不进行裁剪。
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    # 将准备好的数据blob输入网络
    net.setInput(blob)
    # net.getLayerNames(): 这个函数用于获取模型中所有层的名称
    layersNames = net.getLayerNames()
    # 得到模型中未连接的输出层的名称 'yolo_82', 'yolo_94', 'yolo_106'
    # net.getUnconnectedOutLayers()获取的索引是一个二维的数组，并且是1-based的，所以-1来换算成0-based
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    # 前向传播推理获得我们指定层的输出
    # 得到3类数据，outputs[0],outputs[1],outputs[2]，分别为300*85，1200*85，4800*85
    # 85的组成是x,y,w,h,confidence和是80个类别的概率
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv.imshow('Image', img)
    cv.waitKey(1)
