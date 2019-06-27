import numpy as np
import matplotlib.pyplot as plt
myImage = plt.imread('./tree.png')
print(type(myImage))

myImage[:,:,0] = 0
plt.imshow(myImage)
plt.show()
