from PIL import Image
import numpy as np
import os

absolutePath = "afhq_dog"
imgCount = 0
X = np.empty((len(os.listdir(absolutePath)), 4096, 3))
for fileName in os.listdir(absolutePath):
    imagePath = os.path.join(absolutePath, fileName)
    image = np.array(Image.open(imagePath).resize((64, 64), Image.BILINEAR))  # TODO look for warning
    X[imgCount] = image.flatten().reshape((4096, 3))
    imgCount += 1
X_0 = X[:, :, 0]  # red
X_1 = X[:, :, 1]  # green
X_2 = X[:, :, 2]  # blue
