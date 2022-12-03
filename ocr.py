import re
import os
import cv2
import math
import imutils
import datetime
import numpy as np
import pyzbar.pyzbar as pyzbar
from paddleocr import PaddleOCR


class Config:
    def __init__(self):
        pass

    src = "./img/"
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    color_list = ['orange']
    color_num_list = {
        'orange': {'Lower': np.array([11, 43, 46]), 'Upper': np.array([20, 255, 255])},
    }
    color_dist = {
        '绿码': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
        '红码': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
        '黄码': {'Lower': np.array([26, 43, 46]), 'Upper': np.array([34, 255, 255])},
    }
    re_dict = {
        'name': re.compile(r'^\D{0,2}?[\u4e00-\u9fa5]+\s?[(|（][\u4e00-\u9fa5]{2}[)|）]'),
        'scan': re.compile(r'请扫.?场所码'),
        'dates': re.compile(r'\d{4}年\d{1,2}月\d{1,2}日'),
        'real_time': re.compile(r'^\d{1,2}:\d{2}:\d{2}'),
        'in_time': re.compile(r'进场时间[:|：]'),
        'place': re.compile(r'场所名称[:|：]'),
    }


def point_distance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


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


def scan_img():
    nums = len(os.listdir(Config.src))
    print(f'共{nums}张图片，开始信息录入')
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    for filename in os.listdir(Config.src):
        file = Config.src + filename
        result = Config.ocr.ocr(file, cls=True)

        datas = [' ,', ' ,', ' ,', ' ,', ' ,', ' ,', ]

        for i in result:
            for j in i:
                print(j)
                if Config.re_dict['name'].match(j[1][0]):
                    b = j[1][0]
                    for c in range(len(b)):
                        if b[c] == '（' or b[c] == '(':
                            b = b[:c]
                            b += ','
                            break
                    if len(b) == 2 or '*' in b:
                        datas[0] = '姓名未显示,'
                    else:
                        datas[0] = b
                if Config.re_dict['scan'].search(j[1][0]):
                    datas[1] = '场所码未扫描,'
                if Config.re_dict['in_time'].search(j[1][0]):
                    a = j[1][0]
                    b = re.compile(r'\d{4}-\d{1,2}-\d{1,2}').search(a)
                    b = b.group()
                    c = re.compile(r'\d{2}[:|：]\d{2}[:|：]\d{2}').search(a)
                    c = c.group()
                    c = [":" if c[k] == "：" else c[k] for k in range(len(c))]
                    c = ''.join(c)
                    a_t = datetime.datetime.strptime(f'{b} {c}', "%Y-%m-%d %H:%M:%S")
                    if (t - a_t).total_seconds() > 3600:
                        a_t = a_t.strftime("%H:%M:%S") + "(已超时),"
                    else:
                        a_t = a_t.strftime("%H:%M:%S") + ','
                    datas[4] = a_t
                if Config.re_dict['dates'].search(j[1][0]):
                    a = j[1][0]
                    b = re.compile(r'周[\u4e00-\u9fa5]').search(a)
                    b = b.group()
                    a = a[:11] + f'({b})'
                    a += ','
                    datas[3] = a
                if Config.re_dict['place'].search(j[1][0]):
                    a = j[1][0]
                    a = a[5:]
                    a += ','
                    datas[5] = a

        if not datas[1] == '场所码未扫描,':
            img = cv2.imread(file, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar.decode(gray)
            for barcode in barcodes:
                points = []
                for point in barcode.polygon:
                    points.append([point[0], point[1]])
                points = np.array(points, dtype="float32")
                w, h = point_distance(points[0], points[1]), point_distance(points[1], points[2])
                dst_rect = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]],
                    dtype="float32")
                m = cv2.getPerspectiveTransform(points, dst_rect)
                qr = cv2.warpPerspective(img, m, (w, h))
                for i in Config.color_dist.keys():
                    k = detect_color(qr, i)
                    if k:
                        i += ','
                        datas[1] = i
                        break
                break

        orange = []
        for i in Config.color_list:
            img = cv2.imread(file, 1)
            img = imutils.resize(img, width=500)
            gs = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(gs, cv2.COLOR_BGR2HSV)
            erode_hsv = cv2.erode(hsv, None, iterations=2)
            range_hsv = cv2.inRange(erode_hsv, Config.color_num_list[i]['Lower'], Config.color_num_list[i]['Upper'])
            contours, _ = cv2.findContours(range_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=cv2.contourArea, reverse=True)
            contours = contours[:5]
            for index, contour in enumerate(contours):
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box = box.reshape(4, 2)
                src_rect = order_points(box)
                the_area = math.fabs(cv2.contourArea(src_rect))
                if 1000 < the_area:
                    # cv2.imshow('camera', image)
                    # cv2.waitKey(0)
                    orange.append(contour)

        if len(orange) == 0:
            datas[2] = "24h内为绿码,"
        elif len(orange) == 1:
            datas[2] = "24h内为橙码,"
        elif len(orange) == 2:
            datas[2] = "48h内为橙码,"
        elif len(orange) == 3:
            datas[2] = "72h内为橙码,"
        elif len(orange) == 4:
            datas[2] = "6天内为橙码,"
        else:
            datas[2] = "7天内为橙码,"

        if not os.path.exists("result.csv"):
            with open('result.csv', 'w', encoding='utf-8-sig') as f:
                f.write('姓名,健康码状态,核酸检测结果,日期,时间,场所,\n')

        with open('result.csv', 'a+', encoding='utf-8-sig') as f:
            f.write(''.join(datas) + "\n")

        x = os.listdir(Config.src).index(filename) + 1
        print(f'已完成{x}张图片的扫描与录入[{x}/{nums}]')
    print(f'{nums}张图片的信息录入完成，打开result.csv文件查看结果')


if __name__ == '__main__':
    scan_img()
