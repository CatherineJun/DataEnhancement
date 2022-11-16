import numpy as np
import cv2


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    print(len(channels))
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

im = cv2.imread('data/2.jpg')
print(np.shape(im))
im=cv2.resize(im,(800,600))#小于或等于屏幕分辨率

cv2.imshow('im1', im)
cv2.waitKey(0)
eq = hisEqulColor(im)
cv2.imshow('image2',eq )
cv2.waitKey(0)
# cv2.imwrite('lena2.jpg',eq)

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import collections
#
#
# def rgb_hsi(rgb_image):
#     # 保存原始图像的行列数
#     rows = int(rgb_image.shape[0])
#     cols = int(rgb_image.shape[1])
#     # 图像复制
#     hsi_image = rgb_image.copy()
#     # 通道拆分
#     b = rgb_image[:, :, 0]
#     g = rgb_image[:, :, 1]
#     r = rgb_image[:, :, 2]
#     # 归一化到[0,1]
#     b = b / 255.0
#     g = g / 255.0
#     r = r / 255.0
#     for i in range(rows):
#         for j in range(cols):
#             num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
#             den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
#             if den == 0:
#                 hsi_h = 0
#             else:
#                 theta = float(np.arccos(num / den))
#                 if b[i, j] <= g[i, j]:
#                     hsi_h = theta
#                 else:
#                     hsi_h = 2*np.pi - theta
#
#             min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
#             sum = b[i, j]+g[i, j]+r[i, j]
#             if sum == 0:
#                 hsi_s = 0
#             else:
#                 hsi_s = 1 - 3*min_RGB/sum
#
#             hsi_h = hsi_h/(2*np.pi)
#             hsi_i = sum/3.0
#             # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
#             hsi_image[i, j, 0] = hsi_h*255
#             hsi_image[i, j, 1] = hsi_s*255
#             hsi_image[i, j, 2] = hsi_i*255
#     return hsi_image
#
#
# def hsi_rgb(hsi_image):
#     # 保存原始图像的行列数
#     rows = np.shape(hsi_image)[0]
#     cols = np.shape(hsi_image)[1]
#     # 对原始图像进行复制
#     rgb_image = hsi_image.copy()
#     # 对图像进行通道拆分
#     hsi_h = hsi_image[:, :, 0]
#     hsi_s = hsi_image[:, :, 1]
#     hsi_i = hsi_image[:, :, 2]
#     # 把通道归一化到[0,1]
#     hsi_h = hsi_h / 255.0
#     hsi_s = hsi_s / 255.0
#     hsi_i = hsi_i / 255.0
#     B, G, R = hsi_h, hsi_s, hsi_i
#     for i in range(rows):
#         for j in range(cols):
#             hsi_h[i, j] *= 360
#             if 0 <= hsi_h[i, j] < 120:
#                 B = hsi_i[i, j] * (1 - hsi_s[i, j])
#                 R = hsi_i[i, j] * (1 + (hsi_s[i, j] * np.cos(hsi_h[i, j] * np.pi / 180)) / np.cos(
#                     (60 - hsi_h[i, j]) * np.pi / 180))
#                 G = 3 * hsi_i[i, j] - (R + B)
#             elif 120 <= hsi_h[i, j] < 240:
#                 hsi_h[i, j] = hsi_h[i, j] - 120
#                 R = hsi_i[i, j] * (1 - hsi_s[i, j])
#                 G = hsi_i[i, j] * (1 + (hsi_s[i, j] * np.cos(hsi_h[i, j] * np.pi / 180)) / np.cos(
#                     (60 - hsi_h[i, j]) * np.pi / 180))
#                 B = 3 * hsi_i[i, j] - (R + G)
#             elif 240 <= hsi_h[i, j] <= 300:
#                 hsi_h[i, j] = hsi_h[i, j] - 240
#                 G = hsi_i[i, j] * (1 - hsi_s[i, j])
#                 B = hsi_i[i, j] * (1 + (hsi_s[i, j] * np.cos(hsi_h[i, j] * np.pi / 180)) / np.cos(
#                     (60 - hsi_h[i, j]) * np.pi / 180))
#                 R = 3 * hsi_i[i, j] - (G + B)
#             rgb_image[i, j, 0] = B * 255
#             rgb_image[i, j, 1] = G * 255
#             rgb_image[i, j, 2] = R * 255
#     return rgb_image
#
#
# # 计算灰度图的直方图
# def draw_histogram(grayscale):
#     # 对图像进行通道拆分
#     hsi_i = grayscale[:, :, 2]
#     color_key = []
#     color_count = []
#     color_result = []
#     histogram_color = list(hsi_i.ravel())  # 将多维数组转换成一维数组
#     color = dict(collections.Counter(histogram_color))  # 统计图像中每个亮度级出现的次数
#     color = sorted(color.items(), key=lambda item: item[0])  # 根据亮度级大小排序
#     for element in color:
#         key = list(element)[0]
#         count = list(element)[1]
#         color_key.append(key)
#         color_count.append(count)
#     for i in range(0, 256):
#         if i in color_key:
#             num = color_key.index(i)
#             color_result.append(color_count[num])
#         else:
#             color_result.append(0)
#     color_result = np.array(color_result)
#     return color_result
#
#
# def histogram_equalization(histogram_e, lut_e, image_e):
#     sum_temp = 0
#     cf = []
#     for i in histogram_e:
#         sum_temp += i
#         cf.append(sum_temp)
#     for i, v in enumerate(lut_e):
#         lut_e[i] = int(255.0 * (cf[i] / sum_temp) + 0.5)
#     equalization_result = lut_e[image_e]
#     return equalization_result
#
#
# x = []
# for i in range(0, 256):  # 横坐标
#     x.append(i)
#
# # 原图及其直方图
# rgb_image = cv2.imread("data/2.jpg")
# cv2.imshow('rgb', rgb_image)
# histogram = draw_histogram(rgb_image)
# plt.bar(x, histogram)  # 绘制原图直方图
# #plt.savefig('./equalization_color/before_histogram.png')
# plt.show()
#
# # rgb转hsi
# hsi_image = rgb_hsi(rgb_image)
# cv2.imshow('hsi_image', hsi_image)
# # cv2.imwrite('./equalization_color/hsi_result.png', hsi_image)
#
# # hsi在亮度分量上均衡化
# histogram_1 = draw_histogram(hsi_image)
# lut = np.zeros(256, dtype=hsi_image.dtype)  # 创建空的查找表
# result = histogram_equalization(histogram_1, lut, hsi_image)  # 均衡化处理
# cv2.imshow('his_color_image', result)
# # cv2.imwrite('./equalization_color/his_color.png', result)  # 保存均衡化后图片
#
# # hsi转rgb
# # image_equ = cv2.imread(r'.\equalization_color\his_color.png')  # 读取图像
# rgb_result = hsi_rgb(result)
# cv2.imshow('rgb_image', rgb_result)
# # cv2.imwrite('./equalization_color/gbr_result.png', rgb_result)
#
# rgb = cv2.imread(".\equalization_color\gbr_result.png")
# histogram_2 = draw_histogram(rgb)
# plt.bar(x, histogram_2)
# # plt.savefig('./equalization_color/after_histogram.png')
# plt.show()
#
# plt.plot(x, lut)  # 绘制灰度级变换曲线图
# # plt.savefig('./equalization_color/Grayscale_transformation_curve.png')
# plt.show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
#
# img = "data/2.jpg"
#
#
# def def_equalizehist(img, L=256):
#     img = cv2.imread(img, 0)
#     cv2.imshow("ori", img)
#     h, w = img.shape
#     # 计算图像的直方图，即存在的每个灰度值的像素点数量
#     hist = cv2.calcHist([img], [0], None, [256], [0, 255])
#     # 计算灰度值的像素点的概率，除以所有像素点个数，即归一化
#     hist[0:255] = hist[0:255] / (h * w)
#     # 设置Si
#     sum_hist = np.zeros(hist.shape)
#     # 开始计算Si的一部分值，注意i每增大，Si都是对前i个灰度值的分布概率进行累加
#     for i in range(256):
#         sum_hist[i] = sum(hist[0:i + 1])
#     equal_hist = np.zeros(sum_hist.shape)
#     # Si再乘上灰度级，再四舍五入
#     for i in range(256):
#         equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)
#     equal_img = img.copy()
#     # 新图片的创建
#     for i in range(h):
#         for j in range(w):
#             equal_img[i, j] = equal_hist[img[i, j]]
#
#     equal_hist = cv2.calcHist([equal_img], [0], None, [256], [0, 256])
#     equal_hist[0:255] = equal_hist[0:255] / (h * w)
#     cv2.imshow("inverse", equal_img)
#     # 显示最初的直方图
#     # plt.figure("原始图像直方图")
#     plt.plot(hist, color='b')
#     plt.show()
#     # plt.figure("直方均衡化后图像直方图")
#     plt.plot(equal_hist, color='r')
#     plt.show()
#     cv2.waitKey()
#     # return equal_hist
#     return [equal_img, equal_hist]
#
# def_equalizehist(img)


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # histogram equalization
# def hist_equal(img, z_max=255):
#     H, W = img.shape
#     # S is the total of pixels
#     S = H * W * 1.
#
#     out = img.copy()
#
#     sum_h = 0.
#
#     for i in range(1, 255):
#         ind = np.where(img == i)
#         sum_h += len(img[ind])
#         z_prime = z_max / S * sum_h
#         out[ind] = z_prime
#
#     out = out.astype(np.uint8)
#
#     return out
#
#
# # Read image
# img = cv2.imread("data/7.jpg", 0).astype(np.float)
#
# # histogram normalization
# out = hist_equal(img)
#
# # Display histogram
# plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
# plt.savefig("out_his.png")
# plt.show()
#
# # Save result
# cv2.imshow("result", out)
# cv2.imwrite("out.jpg", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()