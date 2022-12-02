import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PCA as p

# TODO check absolute path?
# absolutePath = input('Enter the path location: ')
absolutePath = r'C:\Users\gulce\Desktop\EEE 8TH SEMESTER\CS 464\Homeworks\HW2\afhq_dog'
imgCount = 0
X = np.empty((len(os.listdir(absolutePath)), 4096, 3))
for fileName in os.listdir(absolutePath):
    imagePath = os.path.join(absolutePath, fileName)
    image = np.array(Image.open(imagePath).resize((64, 64), Image.Resampling.BILINEAR))
    X[imgCount] = image.flatten().reshape((4096, 3))
    imgCount += 1
X0 = X[:, :, 0]  # red
X1 = X[:, :, 1]  # green
X2 = X[:, :, 2]  # blue

# TODO sum of PVE 70% --> denominator? (TA)
# TODO PVE graph
pca_X0 = p.PCA()
pca_X1 = p.PCA()
pca_X2 = p.PCA()

pca_X0.getEigen(X0)
eigenValuesL_X0, eigenVectorsL_X0 = pca_X0.getkLargestPCs(10)
listPVE_X0 = pca_X0.getPVE(eigenValuesL_X0)
print(listPVE_X0)

pca_X1.getEigen(X1)
eigenValuesL_X1, eigenVectorsL_X1 = pca_X1.getkLargestPCs(10)
listPVE_X1 = pca_X1.getPVE(eigenValuesL_X1)
print(listPVE_X1)

pca_X2.getEigen(X2)
eigenValuesL_X2, eigenVectorsL_X2 = pca_X2.getkLargestPCs(10)
listPVE_X2 = pca_X2.getPVE(eigenValuesL_X2)
print(listPVE_X2)

normalizedX0_PCs = pca_X0.getNormalizedPCs(eigenVectorsL_X0)
normalizedX1_PCs = pca_X1.getNormalizedPCs(eigenVectorsL_X1)
normalizedX2_PCs = pca_X2.getNormalizedPCs(eigenVectorsL_X2)
normalized_PCs = np.stack((normalizedX0_PCs, normalizedX1_PCs, normalizedX2_PCs), axis=3)
print(normalized_PCs.shape)

# TODO title, label, subplot, etc.
fig, axs = plt.subplots(2, 5)
for i in range(0, 5):
    axs[0, i].imshow(normalized_PCs[i])
for i in range(0, 5):
    axs[1, i].imshow(normalized_PCs[i + 5])


def perform(k, originalImg):
    eigenValuesL_X0, eigenVectorsL_X0 = pca_X0.getkLargestPCs(k)
    eigenValuesL_X1, eigenVectorsL_X1 = pca_X1.getkLargestPCs(k)
    eigenValuesL_X2, eigenVectorsL_X2 = pca_X2.getkLargestPCs(k)

    originalImgX0 = originalImg[:, 0]
    originalImgX1 = originalImg[:, 1]
    originalImgX2 = originalImg[:, 2]

    nOriginalImgX0 = originalImgX0 - originalImgX0.mean(axis=0, keepdims=True)
    nOriginalImgX1 = originalImgX1 - originalImgX1.mean(axis=0, keepdims=True)
    nOriginalImgX2 = originalImgX2 - originalImgX2.mean(axis=0, keepdims=True)

    dotProduct_X0 = (np.dot(np.dot(eigenVectorsL_X0, nOriginalImgX0), eigenVectorsL_X0)
                     + originalImgX0.mean(axis=0, keepdims=True)).reshape((64, 64))
    dotProduct_X1 = (np.dot(np.dot(eigenVectorsL_X1, nOriginalImgX1), eigenVectorsL_X1)
                     + originalImgX1.mean(axis=0, keepdims=True)).reshape((64, 64))
    dotProduct_X2 = (np.dot(np.dot(eigenVectorsL_X2, nOriginalImgX2), eigenVectorsL_X2)
                     + originalImgX2.mean(axis=0, keepdims=True)).reshape((64, 64))

    dotProduct = np.stack((dotProduct_X0, dotProduct_X1, dotProduct_X2), axis=2)
    return dotProduct


imgPath = r'C:\Users\gulce\Desktop\EEE 8TH SEMESTER\CS 464\Homeworks\HW2\afhq_dog\flickr_dog_000002.jpg'
originalImage = Image.open(imgPath)
originalImg = np.array(originalImage.resize((64, 64), Image.Resampling.BILINEAR)).flatten().reshape((4096, 3))
# TODO title, label, subplot, etc.
plt.figure()
plt.imshow(originalImage)


k_list = [1, 50, 250, 500, 1000, 4096]

# TODO title, label, subplot, etc.
fig, axs = plt.subplots(3, 2)
r = 0
c = 0
for k in k_list:
    reconstructedImage = perform(k, originalImg)
    axs[r, c].imshow(reconstructedImage.astype('uint8'))
    r += 1
    if r == 3:
        r = 0
        c = 1
plt.show()