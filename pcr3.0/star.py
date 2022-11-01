import cv2
import glob
import os
import re
import numpy as np
import paddlehub as hub


class Config:
    def __init__(self):
        pass

    src = "./output/2"


def get_star():
    # 文字识别
    results = hub.Module(name="chinese_ocr_db_crnn_mobile").recognize_text(
        images=[cv2.imread(image_path) for image_path in glob.glob(f"{Config.src}/*.jpg")],
        use_gpu=False, output_dir='demo_result',
        visualization=True, box_thresh=0.5, text_thresh=0.5)

    # print(results)

    star = re.compile(r'：[\u4e00-\u9fa5]{2,4}[*]?')
    for result in results:
        data = [i['text'] if (
            star.search(i['text'])
        ) else '' for i in result['data']]
        dates = data.copy()
        for i in dates:
            if i == '':
                data.remove(i)
        wd = star.search(data[0])
        fi = wd.group()[1:]
        print(fi)
        if "*" in fi:
            print("带星号")
        else:
            print("未带星号")


if __name__ == '__main__':
    get_star()
