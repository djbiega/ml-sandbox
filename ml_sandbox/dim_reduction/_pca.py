from typing import Tuple, Optional, List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

FLOAT_ARR = npt.NDArray[np.float64]

class PCA():
    """
    Principal Component Analysis implemented with SVD and eigendecomposition
    """
    def __init__(self, n_components: int, method="svd"):
        if method not in self.supported_methods:
            raise ValueError(f"Input method {method} not supported. Supported methods: {*self.supported_methods,}")
        self.n_components = n_components
        self.method = method
        self._variances = None

    @property
    def supported_methods(self) -> List[str]:
        return ["eigendecomposition", "svd"]

    @property
    def explained_variance(self) -> Optional[FLOAT_ARR]:
        return self._variances

    @explained_variance.setter
    def explained_variance(self, x: FLOAT_ARR) -> None:
        self._variances = x

    def fit_transform(self, data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Interface to project the data into the desired number of dimensions using
        the specified method

        Args:
            data: Input data to project

        Returns:
            projected data
        """
        method_dict = {
            "eigendecomposition": self._fit_transform_eigendecomp,
            "svd": self._fit_transform_svd,
        }
        return method_dict[self.method](data)

    def _fit_transform_svd(self, data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Project the data using svd

        Args:
            data: Input data to project

        Returns:
            projected data
        """
        standardized_data = self._standardize_data(data)
        u, sigma, vh = np.linalg.svd(standardized_data, full_matrices=False)
        u = u[:, :self.n_components]
        sigma = sigma[:self.n_components]
        self.explained_variance = sigma / sigma.sum()
        projections = np.dot(u, np.diag(sigma))
        return projections

    def _fit_transform_eigendecomp(self, data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Project the data using eigendecomposition

        Args:
            data: Input data to project

        Returns:
            projected data
        """    
        standardized_data = self._standardize_data(data)
        covariance = np.cov(standardized_data, rowvar=False)
        eigenvals, eigenvec = self._get_sorted_eigh(covariance)
        self.explained_variance = eigenvals / eigenvals.sum()
        projections = np.dot(data, eigenvec.T)
        return projections

    def _get_sorted_eigh(self, covariance: FLOAT_ARR) -> Tuple[FLOAT_ARR, FLOAT_ARR]:
        """
        Calculate the eigenvalues & eigenvectors, then sort them in descending order

        Args:
            covariance: Covariance matrix of the input data

        Returns:
            Tuple of eigenvalues and eigenvectors, sorted by descending eigenvalues
        """
        eigenvals, eigenvec = np.linalg.eigh(covariance)
        sort_idx = np.argsort(eigenvals)[::-1]
        sorted_eigenvals = eigenvals[sort_idx][:self.n_components]
        sorted_eigenvec = eigenvec[sort_idx][:self.n_components]
        return sorted_eigenvals, sorted_eigenvec


    def _standardize_data(self, train_data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Standardize the data such that each feature has mean 0

        Args:
            train_data: Input training data

        Returns:
            Standardized training data
        """
        means = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0)
        standardized_data = (train_data-means) / (std+1e-6)
        return standardized_data



def main():
    data, targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    
    # Hard-code to 3 for now
    N_COMPONENTS = 3

    pca = PCA(N_COMPONENTS, method="svd")
    projections = pca.fit_transform(data)

    plt.plot(np.arange(len(pca.explained_variance)), pca.explained_variance)
    plt.savefig("explained_variance.png")
    plt.close()

    fig = plt.figure()
    if N_COMPONENTS == 3:
        ax = fig.add_subplot(projection='3d')
    elif N_COMPONENTS == 2:
        ax = fig.add_subplot()
    else:
        return

    for i in np.unique(targets):
        if ax.name == "3d":
            ax.scatter(
                projections[:, 0][targets == i],
                projections[:, 1][targets == i],
                projections[:, 2][targets == i],
                label=i
            )
        else:
            ax.scatter(
                projections[:, 0][targets == i],
                projections[:, 1][targets == i],
                label=i
            )
    plt.legend()
    plt.savefig("pca_projections_svd.png")
    plt.close()
    return

if __name__ == '__main__':
    main()
