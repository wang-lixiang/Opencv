import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('test.mp4')
pTime = 0

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
# 实例化一个脸部识别，0.75是人脸检测的置信度阈值，也就是只有识别度大于0.75时才会返回检测到的人脸
faceDetection = mpFace.FaceDetection(0.75)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        # results.detections返回的值有label_id,score,location_data
        for id, detection in enumerate(results.detections):
            # 接收到返回的人脸矩形的xmin,ymin,width,height
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            # mediapipe返回的是都是比例值，需要将其扩展到实际的图像的值
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(50)
