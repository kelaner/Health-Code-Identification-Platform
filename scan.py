import re
import os
import cv2
import math
import logging
import pyttsx3
import winsound
import numpy as np
import pyzbar.pyzbar as pyzbar
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# 关闭DEBUG和WARNING
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)


class Config:
    def __init__(self):
        pass

    src = "./output/temp/"
    qr_color = []
    color_dist = {
        '绿码': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
        '红码': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
        '黄码': {'Lower': np.array([26, 43, 46]), 'Upper': np.array([34, 255, 255])},
    }
    engine = pyttsx3.init()
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    color_list = ['orange']
    color_num_list = {
        'orange': {'Lower': np.array([0, 43, 46]), 'Upper': np.array([25, 255, 255])},
    }


def get_voice():
    Config.engine.setProperty('rate', 200)  # 设置语速
    Config.engine.setProperty('volume', 2.0)  # 设置音量


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


def judge_card():
    for filename in os.listdir(Config.src):
        winsound.Beep(600, 300)
        img = cv2.imread(Config.src + filename, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)

        if barcodes:
            print("识别到健康码")
            Config.engine.say("识别到健康码")
            Config.engine.runAndWait()
            return "qr", img, barcodes
        else:
            return "star", filename, 0

    return "None", 0, 0


def qr_scan(img, barcodes):
    for barcode in barcodes:
        points = []
        for point in barcode.polygon:
            points.append([point[0], point[1]])
        src_rect = order_points(points)
        points = np.array(points, dtype=np.int32)
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
                a = Config.qr_color.pop()
                print(a)
                Config.engine.say(a)
                Config.engine.runAndWait()
            break
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


def star_scan(filename):
    file = Config.src + filename
    result = Config.ocr.ocr(file, cls=True)
    # for i in result:
    #     for j in i:
    #         print(j)
    star = re.compile(r'[\u4e00-\u9fa5]{7}卡')

    t = 0
    for i in result:
        for line in i:
            if star.search(line[1][0]):
                print("识别到行程卡")
                Config.engine.say("识别到行程卡")
                Config.engine.runAndWait()
                t = 1
                break
    if t == 0:
        return 0
    star = re.compile(r'[\u4e00-\u9fa5]{1,9}[*]?')
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
    return 1


def color_num(image):
    orange = []
    for i in Config.color_list:
        gs = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(gs, cv2.COLOR_BGR2HSV)
        erode_hsv = cv2.erode(hsv, None, iterations=2)
        range_hsv = cv2.inRange(erode_hsv, Config.color_num_list[i]['Lower'], Config.color_num_list[i]['Upper'])
        # cv2.imshow('hsv', range_hsv)
        # cv2.waitKey(0)
        contours, _ = cv2.findContours(range_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        contours = contours[:5]
        for index, contour in enumerate(contours):
            # c = max(contours, key=cv2.contourArea)
            # print('c:',c)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], -1, (255, 100, 100), 2)
            # cv2.imshow('camera', image)
            # cv2.waitKey(0)
            box = box.reshape(4, 2)
            src_rect = order_points(box)
            the_area = math.fabs(cv2.contourArea(src_rect))
            if 1000 < the_area:
                orange.append(contour)
    if len(orange) == 0:
        print("24h内为绿码")
        Config.engine.say("24小时内为绿码")
        Config.engine.runAndWait()
    elif len(orange) == 1:
        print("24h内为橙码")
        Config.engine.say("24小时内为橙码")
        Config.engine.runAndWait()
    elif len(orange) == 2:
        print("48h内为橙码")
        Config.engine.say("48小时内为橙码")
        Config.engine.runAndWait()
    elif len(orange) == 3:
        print("72h内为橙码")
        Config.engine.say("72小时内为橙码")
        Config.engine.runAndWait()
    elif len(orange) == 4:
        print("6天内为橙码")
        Config.engine.say("6天内为橙码")
        Config.engine.runAndWait()
    else:
        print("7天内为橙码")
        Config.engine.say("7天内为橙码")
        Config.engine.runAndWait()


if __name__ == '__main__':
    get_voice()
    # while True:
    flag, a, b = judge_card()
    # print(flag)

    if flag == "qr":
        qr_scan(a, b)
        Config.qr_color = []
        color_num(a)
        clean_temp()
    elif flag == "star":
        star_scan(a)
        clean_temp()
