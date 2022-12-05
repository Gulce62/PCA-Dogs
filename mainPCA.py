import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PCA as p


def plotPVE(listPVE, colorName):
    # Plot the PVE vs PC graph
    plt.figure()
    title = 'PVE vs. Principal Component for ' + colorName
    plt.title(title)
    plt.xlabel('Principal Component')
    plt.ylabel('Portion of Variance Explained')
    x = np.arange(1, len(listPVE)+1, 1)
    plt.plot(x, listPVE, marker='o')


def getKmin(pca):
    # Find the minimum number of PCs that are required to obtain at least 70% PVE
    k = 1
    sumPVE = 0
    while sumPVE < 0.70:
        eigenValuesL, eigenVectorsL = pca.getkLargestPCs(k)
        listPVE = pca.getPVE(eigenValuesL)
        sumPVE = np.sum(listPVE)
        k += 1
    return k


absolutePath = input('Enter the path location: ')
# Check if the OS path of the folder exists
try:
    X = np.empty((len(os.listdir(absolutePath)), 4096, 3))
    print('Folder path is correct!!!')
except Exception:
    print('Invalid folder path!!!')
    sys.exit(1)

imgCount = 0
for fileName in os.listdir(absolutePath):  # read each image in the folder and store them into a numpy array
    imagePath = os.path.join(absolutePath, fileName)
    # Check if the OS path of the images are valid
    try:
        image = Image.open(imagePath)
    except Exception:
        print('Invalid image path!!!')
        sys.exit(1)
    image = np.array(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=int)
    X[imgCount] = image.reshape((4096, 3))
    imgCount += 1
# Split image array into its R, G, and B components
X0 = X[:, :, 0]  # red
X1 = X[:, :, 1]  # green
X2 = X[:, :, 2]  # blue

pca_X0 = p.PCA()  # PCA for red
pca_X1 = p.PCA()  # PCA for green
pca_X2 = p.PCA()  # PCA for blue

pca_X0.getEigen(X0)
eigenValuesL_X0, eigenVectorsL_X0 = pca_X0.getkLargestPCs(10)  # get first 10 eigenvalues and eigenvectors for red
listPVE_X0 = pca_X0.getPVE(eigenValuesL_X0)  # get the list of first 10 PVE for red
print('PVEs of the first 10 PCs for red: ', listPVE_X0)
print('Sum of the first 10 PCs for red: ', np.sum(listPVE_X0))
plotPVE(listPVE_X0, 'Red')

pca_X1.getEigen(X1)
eigenValuesL_X1, eigenVectorsL_X1 = pca_X1.getkLargestPCs(10)  # get 10 eigenvalues and eigenvectors for green
listPVE_X1 = pca_X1.getPVE(eigenValuesL_X1)  # get the list of first 10 PVE for green
print('PVEs of the first 10 PCs for green: ', listPVE_X1)
print('Sum of the first 10 PCs for green: ', np.sum(listPVE_X1))
plotPVE(listPVE_X0, 'Green')

pca_X2.getEigen(X2)
eigenValuesL_X2, eigenVectorsL_X2 = pca_X2.getkLargestPCs(10)  # get 10 eigenvalues and eigenvectors for blue
listPVE_X2 = pca_X2.getPVE(eigenValuesL_X2)  # get the list of first 10 PVE for blue
print('PVEs of the first 10 PCs for blue: ', listPVE_X2)
print('Sum of the first 10 PCs for blue: ', np.sum(listPVE_X2))
plotPVE(listPVE_X0, 'Blue')

normalizedX0_PCs = pca_X0.getNormalizedPCs(eigenVectorsL_X0)
normalizedX1_PCs = pca_X1.getNormalizedPCs(eigenVectorsL_X1)
normalizedX2_PCs = pca_X2.getNormalizedPCs(eigenVectorsL_X2)
normalized_PCs = np.stack((normalizedX0_PCs, normalizedX1_PCs, normalizedX2_PCs), axis=3)

k_X0 = getKmin(pca_X0)
print('The minimum number of PCs that are required to obtain at least 70% PVE for red is', k_X0)
k_X1 = getKmin(pca_X1)
print('The minimum number of PCs that are required to obtain at least 70% PVE for green is', k_X1)
k_X2 = getKmin(pca_X2)
print('The minimum number of PCs that are required to obtain at least 70% PVE for blue is', k_X2)
print('The minimum number of PCs that are required to obtain at least 70% PVE for all channels is',
      max(k_X0, k_X1, k_X2))

fig, axs = plt.subplots(2, 5, figsize=(16, 8))
for i in range(0, 5):
    axs[0, i].title.set_text('PC ' + str(i+1))
    axs[0, i].imshow(normalized_PCs[i])
for i in range(0, 5):
    axs[1, i].title.set_text('PC ' + str(i+6))
    axs[1, i].imshow(normalized_PCs[i + 5])


def perform(k, originalImg):
    # Reconstruct the original image with k principal components

    # Find k eigenvalues and eigenvectors for all channels
    eigenValuesL_X0, eigenVectorsL_X0 = pca_X0.getkLargestPCs(k)
    eigenValuesL_X1, eigenVectorsL_X1 = pca_X1.getkLargestPCs(k)
    eigenValuesL_X2, eigenVectorsL_X2 = pca_X2.getkLargestPCs(k)

    # Arrange original image for all channels
    originalImgX0 = originalImg[:, 0]
    originalImgX1 = originalImg[:, 1]
    originalImgX2 = originalImg[:, 2]

    nOriginalImgX0 = originalImgX0 - originalImgX0.mean(axis=0, keepdims=True)
    nOriginalImgX1 = originalImgX1 - originalImgX1.mean(axis=0, keepdims=True)
    nOriginalImgX2 = originalImgX2 - originalImgX2.mean(axis=0, keepdims=True)

    # Find reconstructed image for all channels
    dotProduct_X0 = (np.dot(np.dot(eigenVectorsL_X0, nOriginalImgX0), eigenVectorsL_X0)
                     + originalImgX0.mean(axis=0, keepdims=True)).reshape((64, 64))
    dotProduct_X1 = (np.dot(np.dot(eigenVectorsL_X1, nOriginalImgX1), eigenVectorsL_X1)
                     + originalImgX1.mean(axis=0, keepdims=True)).reshape((64, 64))
    dotProduct_X2 = (np.dot(np.dot(eigenVectorsL_X2, nOriginalImgX2), eigenVectorsL_X2)
                     + originalImgX2.mean(axis=0, keepdims=True)).reshape((64, 64))

    dotProduct = np.stack((dotProduct_X0, dotProduct_X1, dotProduct_X2), axis=2)
    return dotProduct


imgPath = input('Enter the path location of the image: ')
# Check if the OS path of the image is valid
try:
    originalImage = Image.open(imgPath)
    print('Image path is correct!!!')
except Exception:
    print('Invalid image path!!!')
    sys.exit(1)
# Plot the original image
plt.figure()
plt.title('Original Image')
plt.imshow(originalImage)

originalImg = np.array(originalImage.resize((64, 64), Image.Resampling.BILINEAR))
# Plot the reshaped original image to 64x64 pixels
plt.figure()
plt.title('Original Image in 64x64 pixels')
plt.imshow(originalImg)

originalImg = originalImg.reshape((4096, 3))
k_list = [1, 50, 250, 500, 1000, 4096]  # list of the number of principal components

fig, axs = plt.subplots(3, 2, figsize=(8, 16))
r = 0
c = 0
for k in k_list:
    reconstructedImage = perform(k, originalImg)  # find the reconstructed image for k principal component
    axs[r, c].imshow(reconstructedImage.astype('uint8'))
    axs[r, c].title.set_text('Reconstructed Image with ' + str(k) + ' PCs')
    if c == 1:
        c = 0
        r += 1
    else:
        c += 1
plt.show()
