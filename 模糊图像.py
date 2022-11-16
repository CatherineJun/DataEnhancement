import cv2

"""
1.均值模糊
   格式：cv2.blur(img,(n1,n2))  img表示图片，n1表示x方向卷积核大小，n2表示y方向卷积核大小
"""
# 1.均值模糊
img1 = cv2.imread('pic/2.jpg',1)
img1=cv2.resize(img1,(800,600))
blur = cv2.blur(img1,(6,6))
cv2.imshow('img',img1)
cv2.imshow('blur',blur)
cv2.imwrite("pic/2_blur.jpg",blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
2.中值模糊:取内核区域下所有像素的中值，然后用这个中值替换中心元素
   格式：cv2.medianBlur(img,n1) img表示图片，n1表示卷积核大小,应该是正奇数
"""
# 2.中值模糊
medianblur = cv2.medianBlur(img1,5)
cv2.imshow('medianblur',medianblur)
cv2.imwrite("pic/2_medianblur.jpg",medianblur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
3.高斯模糊：在进行均值模糊的时候，其领域类的每个像素权重是相等的，高斯模糊中，会将中心点的
          权重值加大，远离中心点的值减小，在此基础上计算各个领域内各个像素值不同权重的和
   格式：cv2.Gaussianblur(img,(n1,n2),0) img表示图片，n1表示x方向卷积核大小，n2表示y
                                       方向卷积核大小，数值可以不同，但都应为正奇数,
                                       0代表内核大小取(n1,n2)
"""
# 3.高斯模糊
gaussianblur = cv2.GaussianBlur(img1,(5,5),0)
cv2.imshow('gaussianblur',gaussianblur)
cv2.imwrite("pic/2_gaussianblur.jpg",gaussianblur)
cv2.waitKey(0)
cv2.destroyAllWindows()

