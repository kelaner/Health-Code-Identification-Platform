import os
import logging
from paddleocr import PaddleOCR

# import paddlehub as hub

# 关闭DEBUG和WARNING
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)


class Config:
    def __init__(self):
        pass

    src = "./camera/orange/"

    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    results = []


for filename in os.listdir(Config.src):
    file = Config.src + filename
    # result = hub.Module(name="chinese_ocr_db_crnn_mobile").recognize_text(
    #     images=[cv2.imread(file)],
    #     use_gpu=False, output_dir='demo_result',
    #     visualization=True, box_thresh=0.5, text_thresh=0.5)
    result = Config.ocr.ocr(file, cls=True)
    Config.results.append(result)

for i in Config.results:
    print(Config.results.index(i), ":")
    for j in i:
        for k in j:
            print(k)
