import cv2
import numpy as np

myImage = cv2.imread('./tree.png')

print(type(myImage))
myImage = cv2.resize(myImage, (1000,500))
myImage = cv2.flip(myImage, 0)
myImage = cv2.cvtColor(myImage, cv2.COLOR_RGB2GRAY)

cv2.imshow('Photo', myImage)
cv2.waitKey()

cv2.imwrite('output1.jpg', myImage)
