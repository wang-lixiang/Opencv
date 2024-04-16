import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, staticMode=False, max_num_faces=1, refine_landmarks=False, minDetectionCon=0.5,
                 minTrackingCon=0.5):
        self.staticMode = staticMode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        self.mpDraw = mp.solutions.drawing_utils
        # 创建一个绘制规范
        self.drawSpec = self.mpDraw.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)
        self.mpFaceMesh = mp.solutions.face_mesh
        # 最多检测两张人脸，并且对每张人脸都进行关键点检测
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.max_num_faces, self.refine_landmarks,
                                                 self.minDetectionCon, self.minTrackingCon)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            # faceLms是包含了一张人脸的所有的landmark
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,
                                               self.drawSpec)
                # 输出一张人脸上每个点的坐标
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, _ = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture('test.mp4')
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        # 显示帧率
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(50)


if __name__ == "__main__":
    main()
