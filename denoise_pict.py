import cv2
import matplotlib.pyplot as plt

# 计算邻域非白色个数
from PIL import Image, ImageEnhance


def calculate_noise_count(img_obj, w, h):
    """
    计算邻域非白色的个数
    Args:
        img_obj: img obj
        w: width
        h: height
    Returns:
        count (int)
    """
    count = 0
    width, height, s = img_obj.shape
    for _w_ in [w - 1, w, w + 1]:
        for _h_ in [h - 1, h, h + 1]:
            if _w_ > width - 1:
                continue
            if _h_ > height - 1:
                continue
            if _w_ == w and _h_ == h:
                continue
            if (img_obj[_w_, _h_, 0] < 233) or (img_obj[_w_, _h_, 1] < 233) or (img_obj[_w_, _h_, 2] < 233):
                count += 1
    return count


# k邻域降噪
def operate_img(img, k):
    w, h, s = img.shape
    # 从高度开始遍历
    for _w in range(w):
        # 遍历宽度
        for _h in range(h):
            if _h != 0 and _w != 0 and _w < w - 1 and _h < h - 1:
                if calculate_noise_count(img, _w, _h) < k:
                    img.itemset((_w, _h, 0), 255)
                    img.itemset((_w, _h, 1), 255)
                    img.itemset((_w, _h, 2), 255)

    return img


# 四周置白色
def around_white(img):
    w, h, s = img.shape
    for _w in range(w):
        for _h in range(h):
            if (_w <= 5) or (_h <= 5) or (_w >= w - 5) or (_h >= h - 5):
                img.itemset((_w, _h, 0), 255)
                img.itemset((_w, _h, 1), 255)
                img.itemset((_w, _h, 2), 255)
    return img


# 邻域非同色降噪
def noise_unsome_piexl(img):
    '''
        查找像素点上下左右相邻点的颜色，如果是非白色的非像素点颜色，则填充为白色
    :param img:
    :return:
    '''
    # print(img.shape)
    w, h, s = img.shape
    for _w in range(w):
        for _h in range(h):
            if _h != 0 and _w != 0 and _w < w - 1 and _h < h - 1:  # 剔除顶点、底点
                center_color = img[_w, _h]  # 当前坐标颜色
                # print(center_color)
                top_color = img[_w, _h + 1]
                bottom_color = img[_w, _h - 1]
                left_color = img[_w - 1, _h]
                right_color = img[_w + 1, _h]
                cnt = 0
                if all(top_color == center_color):
                    cnt += 1
                if all(bottom_color == center_color):
                    cnt += 1
                if all(left_color == center_color):
                    cnt += 1
                if all(right_color == center_color):
                    cnt += 1
                if cnt < 1:
                    img.itemset((_w, _h, 0), 255)
                    img.itemset((_w, _h, 1), 255)
                    img.itemset((_w, _h, 2), 255)
    return img


def pro_img(path_path, new_path):  # 传入原图片路径和保存路径 如果不想传路径就传入img及将sharp_img返回
    img = cv2.imread(path_path)  # 改为传入
    # 二值化
    ret, img2 = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    # plt.subplot(221), plt.imshow(img)  # 原始图片
    # plt.subplot(222), plt.imshow(img2)  # 降噪图片
    img2 = operate_img(img2, 3)
    img2 = around_white(img2)
    img2 = noise_unsome_piexl(img2)
    # plt.subplot(223), plt.imshow(img2)  # 降噪图片
    region = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    region = region.convert('RGB')
    region = region.resize((160, 60), Image.ANTIALIAS)
    # 对比度增强
    sharpness = ImageEnhance.Contrast(region)
    sharp_img = sharpness.enhance(3.0)
    sharp_img.save(new_path)  # 保存图片
    # plt.subplot(224), plt.imshow(sharp_img)  # 降噪图片
    # plt.show()
    # return sharp_img  # 不想保存就返回图片信息
