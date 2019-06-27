import cv2
import numpy as np

myImage = cv2.imread('./tree.png')

myImage = cv2.Canny(myImage, 100, 200)

cv2.imshow('Photo', myImage)
cv2.waitKey()

cv2.imwrite('output4.jpg', myImage)
