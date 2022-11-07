import cv2
import os
import pyttsx3
import shutil
import numpy as np
import pyzbar.pyzbar as pyzbar


class Config:
    def __init__(self):
        pass

    src = "./output/temp/"
    # root_dir = "./output/temp/"
    qr_color = []
    color_dist = {
        '绿码': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
        '红码': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
        '黄码': {'Lower': np.array([26, 43, 46]), 'Upper': np.array([34, 255, 255])},
    }
    engine = pyttsx3.init()


def get_voice():
    Config.engine.setProperty('rate', 150)  # 设置语速
    Config.engine.setProperty('volume', 1.0)  # 设置音量


# def change_temp():
#     try:
#         if os.path.exists(Config.root_dir):
#             for i in os.listdir(Config.root_dir):
#                 full_path = os.path.join(Config.root_dir, i)
#                 aim_dir = "./output/1/"
#                 if not os.path.exists(aim_dir):
#                     os.mkdir(aim_dir)
#                 shutil.move(full_path, aim_dir)
#                 break
#     except:
#         pass


def clean_temp():
    for i in os.listdir(Config.src):
        path = os.path.join(Config.src, i)
        os.remove(path)


# 坐标点排序 [top-left, top-right, bottom-right, bottom-left]
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[0]
    rect[2] = pts[2]
    rect[1] = pts[1]
    rect[3] = pts[3]
    return rect


def point_distance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


def read_path(file_path):
    for filename in os.listdir(file_path):
        img = cv2.imread(file_path + filename, 1)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # qr = img[450:650, 150:350]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        for barcode in barcodes:
            points = []
            for point in barcode.polygon:
                points.append([point[0], point[1]])
            src_rect = order_points(points)
            points = np.array(points, dtype=np.int32).reshape(-1, 1)
            # 框出二维码
            # cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            w, h = point_distance(points[0], points[1]), point_distance(points[1], points[2])
            dst_rect = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]],
                dtype="float32")
            m = cv2.getPerspectiveTransform(src_rect, dst_rect)
            qr = cv2.warpPerspective(img, m, (w, h))

            # cv2.imshow("QR", qr)
            # cv2.waitKey(0)

            for i in Config.color_dist.keys():
                k = detect_color(qr, i)
                if k:
                    Config.qr_color.append(i)
            break


def detect_color(image, color):
    try:
        gs = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(gs, cv2.COLOR_BGR2HSV)
        erode_hsv = cv2.erode(hsv, None, iterations=2)
        range_hsv = cv2.inRange(erode_hsv, Config.color_dist[color]['Lower'], Config.color_dist[color]['Upper'])
        contours = cv2.findContours(range_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contours) > 1:
            return True
        else:
            return False

    except:
        return None


def get_info():
    try:
        # 表格填充
        with open('demo_result.csv', 'a+', encoding='utf-8-sig') as f:

            a = Config.qr_color.pop()
            f.write(a + "\n")
            Config.engine.say(a)
            Config.engine.runAndWait()
    except:
        pass


if __name__ == '__main__':
    # clean_temp()
    get_voice()
    # while True:
    # change_temp()
    read_path(Config.src)
    get_info()
    clean_temp()
    Config.qr_color = []
