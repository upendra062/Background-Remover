import cv2
import numpy as np
import skimage
from skimage import exposure
# load image and get dimensions
img = cv2.imread("opencv_frame_7.png")

# convert to hsv
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# threshold using inRange
range1 = (20,80,80)
range2 = (90,255,255)
mask = cv2.inRange(hsv,range1,range2)
mask = 255 - mask

# apply morphology opening to mask
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# antialias mask
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(0,255))

result = img.copy()
result[mask==0] = (255,255,255)

# write result to disk
cv2.imwrite("opencv_frame_7.png", mask)
cv2.imwrite("opencv_frame_7.png", result)

# display it
cv2.imshow("mask", mask)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
