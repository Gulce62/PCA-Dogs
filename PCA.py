import numpy as np


class PCA:
    def __init__(self):
        self.numPC = None
        self.eigenValues = None
        self.eigenVectors = None

    def getEigen(self, X):
        # Get the sorted (descending order) eigenvalues and eigenvectors from a feature array
        normalized = X - X.mean(axis=0, keepdims=True)  # subtract the mean
        covariance = np.cov(normalized, rowvar=False)  # find covariance matrix
        self.eigenValues, self.eigenVectors = np.linalg.eig(covariance)  # find eigenvalues and eigenvectors of the covariance matrix
        eigenIndices = np.argsort(-self.eigenValues)  # find the order of the sorted eigenvalues with their indices
        self.eigenVectors = self.eigenVectors.T  # get the transpose of eigenvectors
        self.eigenValues = self.eigenValues[eigenIndices]  # sort the eigenvalues in descending order
        self.eigenVectors = self.eigenVectors[eigenIndices]  # sort the eigenvectors according to eigenvalues

    def getkLargestPCs(self, k):
        # Get the first k largest eigenvalues and their eigenvectors
        self.numPC = k
        largestEigenValues = self.eigenValues[:self.numPC]
        largestEigenVectors = self.eigenVectors[:self.numPC]
        return largestEigenValues, largestEigenVectors

    def getPVE(self, largestEigenValues):
        # Get the PVE of the principle components
        listPVE = []
        for e in largestEigenValues:
            listPVE.append(e / np.sum(self.eigenValues))
        return listPVE

    def getNormalizedPCs(self, largestEigenVectors):
        # Get the normalized version of the principal components
        X_PCs = np.reshape(largestEigenVectors, (self.numPC, 64, 64))
        normalizedX_PCs = (X_PCs - X_PCs.min()) / (X_PCs.max() - X_PCs.min())
        return normalizedX_PCs
