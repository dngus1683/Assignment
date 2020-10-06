import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open("./Lenna.png")

plt.imshow(image)
plt.show()

image2 = image.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(image2)
plt.show()

image3 = image.transpose(Image.ROTATE_180)
plt.imshow(image3)
plt.show()

image4 = image.resize((int(image.width/2),int(image.height/2)))
plt.imshow(image4)
plt.show()


