# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np

if __name__ == '__main__':
	img = cv2.imread("pic/2.jpg")
	img = cv2.resize(img,(800,600))
	h,w,n = img.shape
	print(h,w,n)   #252 250 3
	#使用numpy构建移动矩阵
	M = np.float32([[1,0,20],[0,1,30]])
    #第一个参数为原图像，第二个参数为移动矩阵，可以自定义，第三个参数为输出图像大小
	res = cv2.warpAffine(img,M,(h,w))
	cv2.imshow("ori",img)
	cv2.imshow("img",res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
