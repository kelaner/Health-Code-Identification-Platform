import cv2
import glob
import os
import paddlehub as hub

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

results = hub.Module(name="chinese_ocr_db_crnn_mobile").recognize_text(
    images=[cv2.imread(image_path) for image_path in glob.glob("img/*.jpg")], use_gpu=True, output_dir='ocr_result',
    visualization=True, box_thresh=0.5, text_thresh=0.5)

with open('检测结果统计.csv', 'w', encoding='utf-8-sig') as f:
    f.write('姓名,结果,检测时间,\n')
    for result in results:
        data = [result['data'][i]['text'] + "," if (
                i == 3
                or "阴性" in result['data'][i]['text']
                or "阳性" in result['data'][i]['text']
                or "2022-" in result['data'][i]['text']) else ''
                for i in range(len(result['data']))]
        print(data)
        f.write(''.join(data) + "\n")
