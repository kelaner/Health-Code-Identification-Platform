import os
import cv2
import math
import imutils
import numpy as np


class Config:
    def __init__(self):
        pass

    src = "./video/demo.mp4"
    resizeRate = 1.0  # 缩放
    min_area = 50000  # 区域面积
    min_contours = 8  # 轮廓
    threshold_thresh = 90  # 分类阈值


# 读取视频
cap = cv2.VideoCapture(Config.src)
# cap = cv2.VideoCapture(0)
w, h = None, None
out = None

# cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
# frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frames_num)

# for i in range(frames_num):
#     ret = cap.grab()
#     if i % fps == 0:
#     if i:
#         ret, frame = cap.retrieve()
#         if frame is None:
#             break
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    # frame = imutils.resize(frame, width=750)
    # frame = imutils.rotate_bound(frame, 90)

    if out is None:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if not os.path.exists("./video/result"):
            os.mkdir("./video/result")
        (h, w) = frame.shape[:2]
        out = cv2.VideoWriter('./video/result/demo.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.medianBlur(gray, 7)
        ret, binary = cv2.threshold(binary, Config.threshold_thresh, 255, cv2.THRESH_BINARY)
        binary = cv2.erode(binary, None, iterations=2)
        binary = cv2.Canny(binary, 0, 60, apertureSize=3)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        contours = contours[:5]


        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect


        def point_distance(a, b):
            return int(np.sqrt(np.sum(np.square(a - b))))


        for index, contour in enumerate(contours):
            # print(index)
            # print("len(contour):", len(contour))
            if len(contour) < Config.min_contours:
                break
            while True:
                if contour is None:
                    break
                if len(contour) < 4:
                    break
                # 获取最小矩形包络
                rect = cv2.minAreaRect(contour)
                # print(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box = box.reshape(4, 2)
                src_rect = order_points(box)
                the_area = math.fabs(cv2.contourArea(src_rect))
                # print(the_area)

                if the_area > Config.min_area:
                    w1, h1 = point_distance(src_rect[0], src_rect[1]), point_distance(src_rect[1], src_rect[2])
                    # print(w, h)
                    if w1 > h1:
                        break

                    cv2.line(frame, (int(src_rect[0][0]), int(src_rect[0][1])),
                             (int(src_rect[1][0]), int(src_rect[1][1])),
                             color=(100, 255, 100), thickness=2)
                    cv2.line(frame, (int(src_rect[2][0]), int(src_rect[2][1])),
                             (int(src_rect[1][0]), int(src_rect[1][1])),
                             color=(100, 255, 100), thickness=2)
                    cv2.line(frame, (int(src_rect[2][0]), int(src_rect[2][1])),
                             (int(src_rect[3][0]), int(src_rect[3][1])),
                             color=(100, 255, 100), thickness=2)
                    cv2.line(frame, (int(src_rect[0][0]), int(src_rect[0][1])),
                             (int(src_rect[3][0]), int(src_rect[3][1])),
                             color=(100, 255, 100), thickness=2)
                break

        # 透视变换
        # dst_rect = np.array([
        #     [0, 0],
        #     [w, 0],
        #     [w, h],
        #     [0, h]],
        #     dtype="float32")
        # M = cv2.getPerspectiveTransform(src_rect, dst_rect)
        # warped = cv2.warpPerspective(frame, M, (w, h))

        output = frame
        out.write(output)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
