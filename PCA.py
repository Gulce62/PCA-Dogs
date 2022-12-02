import numpy as np

class PCA:
    def __init__(self):
        self.numPC = None
        self.eigenValues = None
        self.eigenVectors = None

    def getEigen(self, X):
        normalized = X - X.mean(axis=0, keepdims=True)
        covariance = np.cov(normalized, rowvar=False)
        self.eigenValues, self.eigenVectors = np.linalg.eig(covariance)
        eigenIndices = np.argsort(-self.eigenValues)
        self.eigenVectors = self.eigenVectors.T
        self.eigenValues = self.eigenValues[eigenIndices]
        self.eigenVectors = self.eigenVectors[eigenIndices]

    def getkLargestPCs(self, k):
        self.numPC = k
        largestEigenValues = self.eigenValues[:self.numPC]
        largestEigenVectors = self.eigenVectors[:self.numPC]
        return largestEigenValues, largestEigenVectors

    def getPVE(self, largestEigenValues):
        listPVE = []
        for e in largestEigenValues:
            listPVE.append(e/np.sum(self.eigenValues))
        return listPVE

    def getNormalizedPCs(self, largestEigenVectors):
        X_PCs = np.reshape(largestEigenVectors, (self.numPC, 64, 64))
        normalizedX_PCs = (X_PCs - X_PCs.min()) / (X_PCs.max() - X_PCs.min())
        return normalizedX_PCs