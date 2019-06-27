import cv2
import numpy as np

myImage = cv2.imread('./tree.png')

cv2.rectangle(myImage, pt1=(100,100), pt2=(200,200), color=(0,0,255), thickness=10)

cv2.imshow('Photo', myImage)
cv2.waitKey()


cv2.imwrite('output2.jpg', myImage)
