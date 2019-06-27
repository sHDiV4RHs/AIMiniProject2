import cv2
import numpy as np
import matplotlib.pyplot as plt

myImage = cv2.imread('./tree.png')
print(type(myImage))
print(myImage.shape)

myImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2RGB)
plt.imshow(myImage)
plt.show()
