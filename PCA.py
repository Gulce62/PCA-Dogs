import numpy as np


class PCA:
    def __init__(self, numPC):
        self.numPC = numPC
        self.eigenValues = None
        self.eigenVectors = None

    def getEigen(self, X):
        normalized = X - X.mean(axis=0, keepdims=True)
        covariance = np.cov(normalized, rowvar=False)
        self.eigenValues, self.eigenVectors = np.linalg.eig(covariance)
        eigenIndices = np.argsort(-self.eigenValues)
        self.eigenVectors = self.eigenVectors.T
        largestEigenValues = self.eigenValues[eigenIndices][:self.numPC]
        largestEigenVectors = self.eigenVectors[eigenIndices][:self.numPC]
        return largestEigenValues, largestEigenVectors

    def getPVE(self, largestEigenValues):
        listPVE = []
        for e in largestEigenValues:
            listPVE.append(e/np.sum(self.eigenValues))
        return listPVE

    def getPCs(self, largestEigenVectors):
        X_PCs = np.reshape(largestEigenVectors, (self.numPC, 64, 64))
        normalizedX_PCs = (X_PCs - X_PCs.min()) / (X_PCs.max() - X_PCs.min())
        return normalizedX_PCs


