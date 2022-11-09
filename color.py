import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


class Config:
    def __init__(self):
        pass

    src = "./output/1/demo_2.jpg"


color_list = ['orange', 'green']

color_dist = {
    'orange': {'Lower': np.array([11, 43, 46]), 'Upper': np.array([25, 255, 255])},
    'blue': {'Lower': np.array([100, 43, 46]), 'Upper': np.array([124, 255, 255])},
    'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
    'yellow': {'Lower': np.array([26, 43, 46]), 'Upper': np.array([34, 255, 255])},
    'red': {'Lower': np.array([0, 43, 46]), 'Upper': np.array([0, 255, 255])},
}


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


def point_distance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


def get_five_color(image, camera_image):
    for i in color_list:
        gs = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(gs, cv2.COLOR_BGR2HSV)
        erode_hsv = cv2.erode(hsv, None, iterations=2)
        range_hsv = cv2.inRange(erode_hsv, color_dist[i]['Lower'], color_dist[i]['Upper'])
        contours, _ = cv2.findContours(range_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        contours = contours[:5]

        for index, contour in enumerate(contours):
            # c = max(contours, key=cv2.contourArea)
            # print('c:',c)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(image, [box], -1, (255, 100, 100), 2)
            # cv2.imshow('camera', image)
            # cv2.waitKey(0)
            box = box.reshape(4, 2)
            src_rect = order_points(box)
            the_area = math.fabs(cv2.contourArea(src_rect))

            if the_area > 500:
                w, h = point_distance(src_rect[0], src_rect[1]), point_distance(src_rect[1], src_rect[2])
                print("w,h=%d,%d" % (w, h))

                cv2.line(image, (int(src_rect[0][0]), int(src_rect[0][1])), (int(src_rect[1][0]), int(src_rect[1][1])),
                         color=(255, 100, 100), thickness=2)
                cv2.line(image, (int(src_rect[2][0]), int(src_rect[2][1])), (int(src_rect[1][0]), int(src_rect[1][1])),
                         color=(255, 100, 100), thickness=2)
                cv2.line(image, (int(src_rect[2][0]), int(src_rect[2][1])), (int(src_rect[3][0]), int(src_rect[3][1])),
                         color=(255, 100, 100), thickness=2)
                cv2.line(image, (int(src_rect[0][0]), int(src_rect[0][1])), (int(src_rect[3][0]), int(src_rect[3][1])),
                         color=(255, 100, 100), thickness=2)

                # 透视变换
                dst_rect = np.array([
                    [0, 0],
                    [w-1, 0],
                    [w-1, h-1],
                    [0, h-1]],
                    dtype="float32")
                m = cv2.getPerspectiveTransform(src_rect, dst_rect)
                warped = cv2.warpPerspective(camera_image, m, (w, h))
                # cv2.imshow("demo", warped)
                # cv2.waitKey(0)
                if not os.path.exists("./camera"):
                    os.mkdir("./camera")
                if not os.path.exists(f"./camera/{i}"):
                    os.mkdir(f"./camera/{i}")
                # cv2.imwrite(f"./camera/{i}/{index}.png", warped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                warped = warped[:, :, [2, 1, 0]]
                plt.imshow(warped)
                plt.axis('off')
                plt.savefig(f"./camera/{i}/{index}.png")


if __name__ == '__main__':
    image = cv2.imread(Config.src)
    camera_image = cv2.imread(Config.src)
    get_five_color(image, camera_image)
