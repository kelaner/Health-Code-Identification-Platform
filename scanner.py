import re
import os
import cv2
import logging
import pyttsx3
import winsound

import numpy as np
import pyzbar.pyzbar as pyzbar
from paddleocr import PaddleOCR

# 关闭DEBUG和WARNING
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)


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
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")


def get_voice():
    Config.engine.setProperty('rate', 200)  # 设置语速
    Config.engine.setProperty('volume', 2.0)  # 设置音量


def begin_system():
    Config.engine.say("系统启动")
    Config.engine.runAndWait()


def clean_temp():
    if not os.path.exists(Config.src):
        os.mkdir(Config.src)
    for i in os.listdir(Config.src):
        path = os.path.join(Config.src, i)
        # noinspection PyBroadException
        try:
            os.remove(path)
        except Exception:
            pass


def judge_card(file_path):
    for filename in os.listdir(file_path):
        file = file_path + filename
        winsound.Beep(600, 500)
        result = Config.ocr.ocr(file, cls=True)
        qr = re.compile(r'核酸[\u4e00-\u9fa5]{1,2}结果')
        star = re.compile(r'[\u4e00-\u9fa5]{2}大数据行程卡')
        # for idx in result:
        #     for line in idx:
        #         print(line[1][0])

        for i in result:
            for line in i:
                if qr.match(line[1][0]):
                    return "qr", filename
                elif star.match(line[1][0]):
                    print("识别到行程卡")
                    Config.engine.say("识别到行程卡")
                    Config.engine.runAndWait()
                    return "star", result
    return "None", 0


def qr_scan(filename):
    img = cv2.imread(Config.src + filename, 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(gray)

    for barcode in barcodes:
        points = []
        for point in barcode.polygon:
            points.append([point[0], point[1]])
        src_rect = order_points(points)
        points = np.array(points, dtype=np.int32)
        # 框出二维码
        # cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
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
                get_info()
        break


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


def detect_color(image, color):
    # noinspection PyBroadException
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

    except Exception:
        return None


def star_scan(result):
    star = re.compile(r'[\u4e00-\u9fa5]{2,9}[*]?')
    data = [line[1][0] if (
        star.search(line[1][0])
    ) else '' for i in result for line in i]
    dates = data.copy()
    for i in dates:
        if i == '':
            data.remove(i)
    flag = 0
    for i in data:
        i = star.search(i)
        fi = i.group()
        if "*" in fi:
            # print("带星号")
            print(fi)
            Config.engine.say("行程卡带星号")
            Config.engine.runAndWait()
            flag = 1
            break
    if flag == 0:
        print("未带星号")
        Config.engine.say("行程卡未带星号")
        Config.engine.runAndWait()


def get_info():
    # noinspection PyBroadException
    try:
        # 表格填充
        with open('demo_result.csv', 'a+', encoding='utf-8-sig') as f:
            a = Config.qr_color.pop()
            f.write(a + "\n")
            print("识别到健康码")
            Config.engine.say("识别到健康码")
            Config.engine.runAndWait()
            print(a)
            Config.engine.say(a)
            Config.engine.runAndWait()
    except Exception:
        pass


if __name__ == '__main__':
    # clean_temp()
    get_voice()
    # begin_system()
    # while True:
    flag, f = judge_card(Config.src)
    # print(flag, f)
    if flag == "qr":
        qr_scan(f)
        # clean_temp()
        Config.qr_color = []
    elif flag == "star":
        star_scan(f)
        # clean_temp()
    # else:
        # clean_temp()
