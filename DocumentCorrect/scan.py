from transform import four_point_transform
import cv2, imutils
import imgEnhance

def preProcess(image):
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    grayImage  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    edgedImage = cv2.Canny(gaussImage, 75, 200)

    cnts = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)  # Calculating contour circumference
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    return  screenCnt, ratio

if __name__ == "__main__":

    image = cv2.imread("../data/2.jpg")
    screenCnt, ratio = preProcess(image)
    warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)

    enhancer = imgEnhance.Enhancer()
    enhancedImg = enhancer.gamma(warped,1.63)

    cv2.imshow("org", imutils.resize(image, height=500))
    cv2.imshow("gamma", imutils.resize(enhancedImg, height=500))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
