import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, enable=False, smooth_seg=True, detectionCon=0.5,
                 trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_lanmarks = smooth_landmarks
        self.enable = enable
        self.smooth_seg = smooth_seg
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_lanmarks, self.enable, self.smooth_seg,
                                     self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # 识别到全身并画出来
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    # 找到所有关节点的坐标并记录
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return self.lmList


def main():
    # 这里接入摄像头，可以使用test.mp4测试
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        # 着重标记一个点
        if len(lmList) != 0:
            print(lmList[11])
            cv2.circle(img, (lmList[11][1] - 100, lmList[11][2] + 100), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
