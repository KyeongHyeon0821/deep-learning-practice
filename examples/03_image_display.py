import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./images/puppy_and_cat.jpeg')

plt.imshow(img)
plt.show()