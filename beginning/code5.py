import cv2
import numpy as np

myImage = cv2.imread('./tree.png')

# myImage = cv2.medianBlur(myImage, 5)
myImage = cv2.boxFilter(myImage, -1, (5,5))

# kernel = np.ones((5,5),np.float32)/25
# myImage = cv2.filter2D(myImage,-1,kernel)

cv2.imshow('Photo', myImage)
cv2.waitKey()

cv2.imwrite('output3.jpg', myImage)
