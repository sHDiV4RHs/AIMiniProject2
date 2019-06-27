import cv2
import numpy as np

myImage = cv2.imread('./tree.png')

myImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)

kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
myImage = cv2.filter2D(myImage,-1,kernel)
print(myImage.shape)
cv2.imshow('Photo', myImage)
cv2.waitKey()
