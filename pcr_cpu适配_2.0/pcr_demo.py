import cv2
import glob
import os
import re
import numpy as np
import paddlehub as hub
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

src = "img_demo"
# src = "../pcr3.0/output"

# 颜色判断
color_dist = {
    '红码': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
    '绿码': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
    '黄码': {'Lower': np.array([26, 43, 46]), 'Upper': np.array([34, 255, 255])},
}


def read_path(file_path):
    for filename in os.listdir(file_path):
        img = cv2.imread(file_path + '/' + filename, 1)
        qr = img[450:650, 150:350]

        # 选取区域展示
        img = img[:, :, [2, 1, 0]]
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(qr)
        plt.axis('off')
        plt.show()

        for i in color_dist.keys():
            k = detect_color(qr, i)
            if k:
                qr_color.append(i)


def detect_color(image, color):
    gs = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(gs, cv2.COLOR_BGR2HSV)
    erode_hsv = cv2.erode(hsv, None, iterations=2)
    range_hsv = cv2.inRange(erode_hsv, color_dist[color]['Lower'], color_dist[color]['Upper'])
    contours = cv2.findContours(range_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) > 0:
        return True
    else:
        return False


qr_color = []
qr_color_1, qr_color_2, qr_color_3, qr_color_4, qr_color_5 = [], [], [], [], []
read_path(src)

# 文字识别
results = hub.Module(name="chinese_ocr_db_crnn_mobile").recognize_text(
    images=[cv2.imread(image_path) for image_path in glob.glob(f"{src}/*.jpg")],
    use_gpu=False, output_dir='demo_result',
    visualization=True, box_thresh=0.5, text_thresh=0.5)

# 表格填充
with open('demo_result.csv', 'w', encoding='utf-8-sig') as f:
    f.write('姓名,时间,检测日期,场所,健康码状态,24h,48h,72h,6天内,7天内\n')
    name = re.compile(r'[\u4e00-\u9fa5]{2,4}（[\u4e00-\u9fa5]{2}）')
    time = re.compile(r'[0-9]{2}:[0-9]{2}:[0-9]{2}')
    dates = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')
    place = re.compile(r'[\u4e00-\u9fa5]{4}：[\u4e00-\u9fa5]{2,}')
    for result in results:
        data = [i['text'] + "," if (
                name.match(i['text'])
                or time.match(i['text'])
                or dates.search(i['text'])
                or place.match(i['text'])) else ''
                for i in result['data']]
        date = data.copy()
        for i in date:
            if i == '':
                data.remove(i)
        for i in range(len(data[0])):
            if data[0][i] == '（':
                data[0] = data[0][:i]
                data[0] += ','
                break
        # print(data)
        data[2] = data[2][5:15]
        data[2] += ','
        data[3] = data[3][5:]
        k = results.index(result)
        data.append(qr_color[k] + ',')
        # data.append(qr_color_1[k] + ',')
        # data.append(qr_color_2[k] + ',')
        # data.append(qr_color_3[k] + ',')
        # data.append(qr_color_4[k] + ',')
        # data.append(qr_color_5[k] + ',')
        # print(data)
        f.write(''.join(data) + "\n")
