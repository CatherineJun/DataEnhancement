import numpy as np
import cv2 as cv


def binary(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    return binary

def  Transformation(img,src):
    coords=np.column_stack(np.where(img>0))
    print(coords)
    print(coords.shape)
    angle =cv.minAreaRect(coords)[-1]    #最小外接矩形

    print(angle)
    if angle<-45:
        angle = -(90+angle)
    else:
        angle=-angle
    center = (w//2,h//2)
    M= cv.getRotationMatrix2D(center,angle,1.0) #传入中心和角度 得到旋转矩形
    rotated = cv.warpAffine(src,M,(w,h),flags=cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE)
    cv.putText(rotated,'Angle:{:.2f} degrees'.format(angle),(10,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)#绘制文字
    cv.imshow('Rotated',rotated)


src= cv.imread('4.jpg')
(h,w)=src.shape[:2]
gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
binary=binary(src)
Transformation(binary,src)
cv.imshow('src',src)
cv.imshow('binary',binary)
cv.waitKey(0)



