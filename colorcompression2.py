import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
import numpy as np
import cv2

flower = cv2.imread('rope.jpg', cv2.IMREAD_COLOR)
plt.imshow(cv2.cvtColor(flower, cv2.COLOR_BGR2RGB))

data = flower / 255.0
data = data.reshape((-1, 3)) #-1 is called as unknown dimension
#print(data.shape)

#cv.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
#cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
#10 is max_iter
#1.0 is epsilon
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#flags : This flag is used to specify how initial centers are taken
flags = cv2.KMEANS_RANDOM_CENTERS
data = data.astype(np.float32)
compactness, labels, centers = cv2.kmeans(data, 16, None, criteria, 10, flags)

new_colors = centers[labels].reshape((-1, 3))
flower_recolored = new_colors.reshape(flower.shape)

plt.imshow(cv2.cvtColor(flower_recolored, cv2.COLOR_BGR2RGB))
plt.title('16-color Image', size = 16)
plt.show()