import os
import cv2
import math
import imutils
import datetime
import numpy as np


class Config:
    def __init__(self):
        pass

    src = "./video/demo.mp4"
    resizeRate = 1.0  # 缩放
    min_area = 50000  # 区域面积
    min_contours = 100  # 轮廓
    threshold_thresh = 85  # 分类阈值


# 坐标点排序 [top-left, top-right, bottom-right, bottom-left]
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 求两点间的距离
def point_distance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


# 获取最小矩形包络
def get_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = box.reshape(4, 2)
    src_rect = order_points(box)
    the_area = math.fabs(cv2.contourArea(src_rect))
    return src_rect, the_area


# 框取目标范围
def draw_area(frame, src_rect):
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


# 透视变换
def get_warped(w, h, src_rect, frame):
    dst_rect = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]],
        dtype="float32")
    m = cv2.getPerspectiveTransform(src_rect, dst_rect)
    warped = cv2.warpPerspective(frame, m, (w, h))
    return warped


# CV识别
def get_shape(cap):
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=750)
        frame = imutils.rotate_bound(frame, 90)  # 翻转操作

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = cv2.medianBlur(gray, 7)
            ret, binary = cv2.threshold(binary, Config.threshold_thresh, 255, cv2.THRESH_BINARY)
            binary = cv2.erode(binary, None, iterations=2)
            binary = cv2.Canny(binary, 0, 60, apertureSize=3)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=cv2.contourArea, reverse=True)
            contours = contours[:1]

            for index, contour in enumerate(contours):
                while True:
                    if contour is None:
                        break
                    if len(contour) < 4:
                        break
                    src_rect, the_area = get_rect(contour)

                    if the_area > Config.min_area:
                        w, h = point_distance(src_rect[0], src_rect[1]), point_distance(src_rect[1], src_rect[2])
                        if w > h:
                            break
                        photo = frame.copy()
                        draw_area(frame, src_rect)
                        warped = get_warped(w, h, src_rect, photo)
                        if not os.path.exists("./output/temp"):
                            os.mkdir("./output/temp")
                        time = int(datetime.datetime.now().strftime('%H%M%S'))
                        temp_i = f"{index}_{time}.jpg"
                        cv2.imshow("warped", warped)
                        cv2.imwrite(f"./output/temp/{temp_i}", warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    break

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # 读取视频
    cap = cv2.VideoCapture(Config.src)  # 读取本地测试视频
    # cap = cv2.VideoCapture(0)  # 调用摄像设备
    # video = "http://192.168.149.254:4747/video"
    # video = "http://192.168.43.38:11311"
    # cap = cv2.VideoCapture(video)

    # CV识别
    get_shape(cap)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
