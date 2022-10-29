import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


# 配置数据
class Config:
    def __init__(self):
        pass

    src = './img/demo.jpg'
    resizeRate = 0.5  # 缩放
    min_area = 5000  # 区域面积
    min_contours = 8  # 轮廓
    threshold_thresh = 70  # 分类阈值
    epsilon_start = 10  # 角点
    epsilon_step = 10


# 预处理转为灰度图
image = cv2.imread(Config.src)
srcWidth, srcHeight, channels = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 中值滤波平滑，消除噪声
binary = cv2.medianBlur(gray, 7)

# 转换为二值图像
ret, binary = cv2.threshold(binary, Config.threshold_thresh, 255, cv2.THRESH_BINARY)
# cv2.imshow("binary", binary)

plt.subplot(1, 4, 2)
plt.title("binary")
plt.imshow(binary)
plt.axis('off')

# 腐蚀
binary = cv2.erode(binary, None, iterations=2)
# canny 边缘检测
binary = cv2.Canny(binary, 0, 60, apertureSize=3)
# cv2.imshow("Canny", binary)

plt.subplot(1, 4, 3)
plt.title("Canny")
plt.imshow(binary)
plt.axis('off')

# 提取轮廓后，拟合外接多边形（矩形）,轮廓升序排列
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("the count of contours is  %d \n" % (len(contours)))
contours.sort(key=cv2.contourArea, reverse=True)


# for a in range(len(contours)):
#     for b in range(len(contours[a])):
#         for c in range(len(contours[a][b])):
#             cv2.circle(image, (int(contours[a][b][c][0]), int(contours[a][b][c][1])), 5, (255, 0, 0), -1)


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


# 找出外接四边形, c是轮廓的坐标数组
def bounding_box(idx, c):
    if len(c) < Config.min_contours:
        return None, None
    epsilon = Config.epsilon_start
    while True:
        approx = cv2.approxPolyDP(c, epsilon, True)
        print("approx:", approx)
        the_area = math.fabs(cv2.contourArea(approx))
        print(f"idx={idx},epsilon={epsilon},approx={len(approx)},c={len(c)},area={the_area}")
        if len(approx) < 4:
            return None, None
        if the_area > Config.min_area:
            if len(approx) > 4:
                epsilon += Config.epsilon_step
                print("epsilon=%d, count=%d" % (epsilon, len(approx)))
                continue
            else:
                approx = approx.reshape((4, 2))
                # 点重排序, [top-left, top-right, bottom-right, bottom-left]
                src_rect = order_points(approx)
                print("src_rect:", src_rect)

                cv2.line(image, (int(src_rect[0][0]), int(src_rect[0][1])), (int(src_rect[1][0]), int(src_rect[1][1])),
                         color=(100, 255, 100), thickness=2)
                cv2.line(image, (int(src_rect[2][0]), int(src_rect[2][1])), (int(src_rect[1][0]), int(src_rect[1][1])),
                         color=(100, 255, 100), thickness=2)
                cv2.line(image, (int(src_rect[2][0]), int(src_rect[2][1])), (int(src_rect[3][0]), int(src_rect[3][1])),
                         color=(100, 255, 100), thickness=2)
                cv2.line(image, (int(src_rect[0][0]), int(src_rect[0][1])), (int(src_rect[3][0]), int(src_rect[3][1])),
                         color=(100, 255, 100), thickness=2)
                return approx, src_rect
        else:
            print("failed to find boundingBox,idx = %d" % idx)
            return None, None


for idx, c in enumerate(contours):
    # 确认外接矩形
    approx, src_rect = bounding_box(idx, c)
    if approx is None:
        continue
    # 获取最小矩形包络
    rect = cv2.minAreaRect(approx)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = box.reshape(4, 2)
    box = order_points(box)
    print("approx->box")
    print(approx)
    print(src_rect)
    print(box)
    w, h = point_distance(box[0], box[1]), point_distance(box[1], box[2])
    print("w,h=%d,%d" % (w, h))

    # 透视变换
    dst_rect = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]],
        dtype="float32")
    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    # cv2.imwrite("./output/piece%d.png" % idx, warped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    image = image[:, :, [2, 1, 0]]
    plt.subplot(1, 4, 1)
    plt.title("image")
    plt.imshow(image)
    plt.axis('off')
    warped = warped[:, :, [2, 1, 0]]
    plt.subplot(1, 4, 4)
    plt.title("output")
    plt.imshow(warped)
    plt.axis('off')
    plt.show()
    break

print('over')
cv2.waitKey(0)
