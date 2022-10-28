import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


# 配置数据
class Config:
    def __init__(self):
        pass

    src = './img/demo.jpg'
    resizeRate = 1.0  # 缩放
    min_area = 50000  # 区域面积
    min_contours = 8  # 轮廓
    threshold_thresh = 70  # 分类阈值


# 预处理转为灰度图
image = cv2.imread(Config.src)
srcWidth, srcHeight, channels = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 中值滤波平滑，消除噪声
binary = cv2.medianBlur(gray, 7)

# 转换为二值图像
ret, binary = cv2.threshold(binary, Config.threshold_thresh, 255, cv2.THRESH_BINARY)
# cv2.imshow("binary", binary)

# 腐蚀
binary = cv2.erode(binary, None, iterations=2)
# canny 边缘检测
binary = cv2.Canny(binary, 0, 60, apertureSize=3)
# cv2.imshow("Canny", binary)

# 绘制边缘检测结果
plt.subplot(1, 3, 2)
plt.title("binary")
plt.imshow(binary)
plt.axis('off')

# 提取轮廓后，拟合外接多边形（矩形）,轮廓升序排列
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("the count of contours is  %d " % (len(contours)))
contours.sort(key=cv2.contourArea, reverse=True)

# 在原图上显示轮廓点
image_show = image[:, :, [2, 1, 0]]
image_show = np.ascontiguousarray(image_show)
for a in range(len(contours)):
    for b in range(len(contours[a])):
        for c in range(len(contours[a][b])):
            cv2.circle(image_show, (int(contours[a][b][c][0]), int(contours[a][b][c][1])), 1, (0, 0, 255), -1)


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


for index, contour in enumerate(contours):
    if len(contour) < Config.min_contours:
        break
    while True:
        if contour is None:
            break
        if len(contour) < 4:
            break
        # 获取最小矩形包络
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = box.reshape(4, 2)
        src_rect = order_points(box)
        the_area = math.fabs(cv2.contourArea(src_rect))
        print(f"idx={index},contour={len(contour)},area={the_area}")

        if the_area > Config.min_area:
            w, h = point_distance(src_rect[0], src_rect[1]), point_distance(src_rect[1], src_rect[2])
            print("w,h=%d,%d" % (w, h))
            if w > h:
                break

            cv2.line(image_show, (int(src_rect[0][0]), int(src_rect[0][1])), (int(src_rect[1][0]), int(src_rect[1][1])),
                     color=(100, 255, 100), thickness=2)
            cv2.line(image_show, (int(src_rect[2][0]), int(src_rect[2][1])), (int(src_rect[1][0]), int(src_rect[1][1])),
                     color=(100, 255, 100), thickness=2)
            cv2.line(image_show, (int(src_rect[2][0]), int(src_rect[2][1])), (int(src_rect[3][0]), int(src_rect[3][1])),
                     color=(100, 255, 100), thickness=2)
            cv2.line(image_show, (int(src_rect[0][0]), int(src_rect[0][1])), (int(src_rect[3][0]), int(src_rect[3][1])),
                     color=(100, 255, 100), thickness=2)

            # 透视变换
            dst_rect = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]],
                dtype="float32")
            M = cv2.getPerspectiveTransform(src_rect, dst_rect)
            warped = cv2.warpPerspective(image, M, (w, h))
            print(233)
            cv2.imwrite(f"./output_demo/{index}.jpg", warped)
            print(123)

            # 测试output
            plt.subplot(1, 3, 1)
            plt.title("image")
            plt.imshow(image_show)
            plt.axis('off')
            warped = warped[:, :, [2, 1, 0]]
            plt.subplot(1, 3, 3)
            plt.title("output")
            plt.imshow(warped)
            plt.axis('off')
            plt.show()
            break
        else:
            print(f"index={index} failed to find bounding_box")
            break

print('over')
