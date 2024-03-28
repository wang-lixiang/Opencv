import cv2
import numpy as np

# 检测对象的阈值
thres = 0.45
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
# 3表示设置宽度，4表示设置高度，10表示设置亮度
cap.set(3, 500)
cap.set(4, 500)
cap.set(10, 150)

# 读取物品种类形成列表
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    # rstrip('\n') 方法删除末尾的换行符，最后使用 split('\n') 方法将字符串按换行符分割成一个列表
    classNames = f.read().rstrip('\n').split('\n')

# 配置文件包含了模型的结构、层次和参数
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# 已经训练好的参数
weightsPath = 'frozen_inference_graph.pb'

# 初始化了一个目标检测模型
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
# 缩放因子为 1.0 / 127.5。这个因子用于对输入图像进行归一化处理
net.setInputScale(1.0 / 127.5)
# 对输入图像进行零中心化处理
net.setInputMean((127.5, 127.5, 127.5))
# 交换红蓝通道，这通常是因为 OpenCV 加载的图像通道顺序是 BGR，而某些模型的训练数据使用的是 RGB 通道顺序
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    # classIds 包含检测到的对象的类别标签，confs 包含每个检测对象的置信度，bbox 包含每个检测对象的边界框坐标，返回的都是一个二维数组
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds)
    bbox = list(bbox)
    # confs 出来的是一个二维数组，每个对象的置信度作为一行
    # 转换为一维列表，后面的[0]是因为前面转换完是array([0.71197456], dtype=float32)，需去第一个元素
    confs = list(np.array(confs).reshape(1, -1)[0])
    # map函数遍历 confs 中的每个元素，并将每个元素转换为浮点数类型
    confs = list(map(float, confs))
    # 使用非最大抑制算法（NMS）对边界框进行过滤，以去除重叠的边界框，并根据置信度阈值和 NMS 阈值过滤掉低置信度的检测结果
    # 过滤后的边界框索引 indices，二维数组
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
        # classIds是一个二维数组，而且表示的值是从1开始的，所以数组上要减1，表示从0开始
        # 书写种类
        cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        # 书写置信度
        cv2.putText(img, str(round(confs[i] * 100, 2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
