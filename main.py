import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PCA as p

# absolutePath = input('Enter the path location: ')
absolutePath = r'C:\Users\gulce\Desktop\EEE 8TH SEMESTER\CS 464\Homeworks\HW2\afhq_dog'
imgCount = 0
X = np.empty((len(os.listdir(absolutePath)), 4096, 3))
for fileName in os.listdir(absolutePath):
    imagePath = os.path.join(absolutePath, fileName)
    image = np.array(Image.open(imagePath).resize((64, 64), Image.Resampling.BILINEAR), dtype=int)
    X[imgCount] = image.flatten().reshape((4096, 3))
    imgCount += 1
X0 = X[:, :, 0]  # red
X1 = X[:, :, 1]  # green
X2 = X[:, :, 2]  # blue

# TODO sum of PVE 70% --> denominator? (TA)
# TODO PVE graph
pca = p.PCA(10)

eigenValuesL_X0, eigenVectorsL_X0 = pca.getEigen(X0)
listPVE_X0 = pca.getPVE(eigenValuesL_X0)
print(listPVE_X0)

eigenValuesL_X1, eigenVectorsL_X1 = pca.getEigen(X1)
listPVE_X1 = pca.getPVE(eigenValuesL_X1)
print(listPVE_X1)

eigenValuesL_X2, eigenVectorsL_X2 = pca.getEigen(X2)
listPVE_X2 = pca.getPVE(eigenValuesL_X2)
print(listPVE_X2)

normalizedX0_PCs = pca.getPCs(eigenVectorsL_X0)
normalizedX1_PCs = pca.getPCs(eigenVectorsL_X1)
normalizedX2_PCs = pca.getPCs(eigenVectorsL_X2)
normalized_PCs = np.stack((normalizedX0_PCs, normalizedX1_PCs, normalizedX2_PCs), axis=3)
print(normalized_PCs.shape)

# TODO title, label, subplot, etc.
fig, axs = plt.subplots(2, 5)
for i in range(0, 5):
    axs[0, i].imshow(normalized_PCs[i])
for i in range(0, 5):
    axs[1, i].imshow(normalized_PCs[i + 5])


def perform(k, originalImg, X0, X1, X2):
    pca = p.PCA(k)

    eigenValuesL_X0, eigenVectorsL_X0 = pca.getEigen(X0)
    eigenValuesL_X1, eigenVectorsL_X1 = pca.getEigen(X1)
    eigenValuesL_X2, eigenVectorsL_X2 = pca.getEigen(X2)

    # TODO mean? --> where do I add?
    # TODO correct dot products?
    dotProduct_X0 = (np.dot(eigenVectorsL_X0.T, np.dot(eigenVectorsL_X0, originalImg[:, 0])) +
                     X0.mean(axis=0, keepdims=True)).reshape((64, 64))
    dotProduct_X1 = (np.dot(eigenVectorsL_X1.T, np.dot(eigenVectorsL_X1, originalImg[:, 1])) +
                     X1.mean(axis=0, keepdims=True)).reshape((64, 64))
    dotProduct_X2 = (np.dot(eigenVectorsL_X2.T, np.dot(eigenVectorsL_X2, originalImg[:, 2])) +
                     X2.mean(axis=0, keepdims=True)).reshape((64, 64))

    dotProduct = np.stack((dotProduct_X0, dotProduct_X1, dotProduct_X2), axis=2)
    normalizedDP = (dotProduct - dotProduct.min()) / (dotProduct.max() - dotProduct.min())
    return normalizedDP


originalImagePath = r'C:\Users\gulce\Desktop\EEE 8TH SEMESTER\CS 464\Homeworks\HW2\afhq_dog\flickr_dog_000002.jpg'
originalImage = Image.open(originalImagePath)
originalImg = np.array(originalImage.resize((64, 64), Image.Resampling.BILINEAR)).flatten().reshape((4096, 3))
# TODO title, label, subplot, etc.
plt.imshow(originalImage)
plt.imshow(originalImg)

k_list = [1, 50, 250, 500, 1000, 4096]

# TODO title, label, subplot, etc.
for k in k_list:
    normalizedDP = perform(k, originalImg, X0, X1, X2)
    plt.figure()
    plt.imshow(normalizedDP)
plt.show()
