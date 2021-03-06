from typing import Tuple, List, Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

FLOAT_ARR = npt.NDArray[np.float64]

def squared_mahalanobis_distance(
        points: FLOAT_ARR, mean: FLOAT_ARR, covariance: FLOAT_ARR
    ) -> FLOAT_ARR:
    """
    Compute the squared mahalanobis distance

    Args:
        points: input points
        mean: mean of the distribution you are computing the distance from
        covariance: covariance of the distribution you are computing the distance from

    Returns:
        Squared mahalanobis distance
    """
    identity = np.eye(len(covariance))
    inv_covariance = sp.linalg.solve(covariance, identity, assume_a='pos')
    return np.matmul(np.matmul((points-mean).T, inv_covariance), (points-mean))

class MultivariateGaussian():
    """
    Multivariate Gaussian class
    """
    def __init__(self, mean: FLOAT_ARR, covariance: FLOAT_ARR) -> None:
        self.mean = mean
        self.covariance = covariance

    def likelihood(self, x: FLOAT_ARR) -> FLOAT_ARR:
        """
        Return the likelihood of the input being part of this Gaussian.
        For numerical convenice, this computation takes place in the log-scale
        and is then re-converted back

        Args:
            x: input points

        Returns:
            probability
        """
        return np.exp(self._log_likelihood(x))

    def _log_likelihood(self, x: FLOAT_ARR) -> FLOAT_ARR:
        """
        Return the log probability of the input being part of this Gaussian.

        Args:
            x: input points

        Returns:
            log probability
        """

        dims = self.mean.shape[-1]
        sq_mahalanobis_distance = squared_mahalanobis_distance(x, self.mean, self.covariance)
        return -0.5 * (np.log((2*np.pi)**dims * np.linalg.det(self.covariance)) + sq_mahalanobis_distance)

class GMM():
    """
    Gaussian Mixture Model implementation based on the algorithm presented in Section 9.3
    of Pattern Recogntion and Machine Learning by Christopher Bishop
    """
    def __init__(self, num_gaussians: int, convergence_threshold: float=1e-6, 
            covariance_reg: float=1e-6, init: Optional[str]=None
        ) -> None:
        self.num_gaussians = num_gaussians
        self.means = np.zeros(num_gaussians)
        self.mixture_coeffs = np.zeros(num_gaussians)
        self.covariances = np.zeros(num_gaussians)
        self.covariance_reg = covariance_reg
        self.convergence_threshold=convergence_threshold
        self._init = init
        self._log_likelihood_per_iter = []


    @property
    def log_likelihood_per_iter(self):
        return self._log_likelihood_per_iter

    def fit(self, train_data: FLOAT_ARR) -> None:
        """
        Train the GMM

        Args:
            train_data: Input training data
        """
        prev_log_likelihood = 1e6
        count = 0

        self.means, self.covariances, self.mixture_coeffs = self._initialize_gmm(train_data)
        new_log_likelihood = self._log_likelihood(train_data)
        while np.abs(new_log_likelihood - prev_log_likelihood) > self.convergence_threshold:
            difference = np.abs(new_log_likelihood - prev_log_likelihood)
            print(f"=====Iteration {count}: Log Likelihood: {prev_log_likelihood}, Diff:{difference}=====")
            responsibilities = self._expectation_step(train_data)
            self.means, self.covariances, self.mixture_coeffs = self._maximization_step(train_data, responsibilities)
            prev_log_likelihood = new_log_likelihood
            new_log_likelihood = self._log_likelihood(train_data)
            self._log_likelihood_per_iter.append(new_log_likelihood)
            count += 1

    def predict(self, test_data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Make a prediction on input data. This assumes that the model has previously
        been trained

        Args:
            test_data: Input data to make predictions on

        Return:
            classification label for which gaussian the test data belongs to
        """
        predictions = []
        for x in test_data:
            gaussian_likelihoods = []
            for j in range(self.num_gaussians):
                gaussian = MultivariateGaussian(self.means[j], self.covariances[j])
                gaussian_likelihoods.append(self.mixture_coeffs[j] * gaussian.likelihood(x))
            predictions.append(np.argmax(gaussian_likelihoods))
        return np.array(predictions)

    def _expectation_step(self, train_data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Evaluate the responsibilities using the current parameter values

        Args:
            train_data: input training data
        """
        responsibilities = np.zeros((train_data.shape[0], self.num_gaussians))
        for i, x in enumerate(train_data):
            gaussian_likelihoods = []
            for j in range(self.num_gaussians):
                gaussian = MultivariateGaussian(self.means[j], self.covariances[j])
                gaussian_likelihoods.append(self.mixture_coeffs[j] * gaussian.likelihood(x))
            gaussian_likelihoods = np.array(gaussian_likelihoods)
            responsibilities[i] = gaussian_likelihoods / np.sum(gaussian_likelihoods, axis=0)
        return responsibilities

    def _maximization_step(self, train_data: FLOAT_ARR, responsibilities: FLOAT_ARR) -> Tuple[FLOAT_ARR, FLOAT_ARR, FLOAT_ARR]:
        """
        Re-estimate the parameters using the current responsibilities

        Args:
            train_data: input training data
            responsibilities: responsibilities computed from the E-step

        Returns:
            Tuple of means, covariances, and mixture coefficients
        """
        means = self._update_means(train_data, responsibilities)
        covariances = self._update_covariances(train_data, means, responsibilities)
        mixture_coeffs = self._update_mixing_coefficients(responsibilities)
        return means, covariances, mixture_coeffs

    def _log_likelihood(self, train_data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Compute the log-likelihood

        Args:
            train_data: input training data

        Returns:
            Log-likelihood
        """
        likelihood = []
        for x in train_data:
            for j in range(self.num_gaussians):
                gaussian = MultivariateGaussian(self.means[j], self.covariances[j])
                gaussian_likelihood = self.mixture_coeffs[j] * gaussian.likelihood(x)
                likelihood.append(gaussian_likelihood)
        return np.sum(np.log(likelihood))

    def _update_means(self, train_data: FLOAT_ARR, responsibilities: FLOAT_ARR) -> FLOAT_ARR:
        """
        Helper function to update the means as part of the M step

        Args:
            train_data: input training data
            responsibilities: responsibilities computed from the E-step

        returns:
            Updated mean value
        """
        points_per_cluster = responsibilities.sum(axis=0)
        means = []
        for i  in range(self.num_gaussians):
            means.append(1/points_per_cluster[i] * np.sum(responsibilities[:, i][:, None]*train_data, axis=0))
        return np.array(means)

    def _update_covariances(self, train_data: FLOAT_ARR, updated_means: FLOAT_ARR, responsibilities: FLOAT_ARR) -> FLOAT_ARR:
        """
        Helper function to update the covariances as part of the M step

        Args:
            train_data: input training data
            updated_means: means which have already been updated as part of this M step
            responsibilities: responsibilities computed from the E-step

        Returns:
            Updated covariance values
        """
        identity_cov_reg = np.eye(updated_means.shape[-1])*self.covariance_reg
        points_per_cluster = responsibilities.sum(axis=0)
        covariances = []
        for i  in range(self.num_gaussians):
            new_cov = np.matmul((responsibilities[:, i][:, None]*(train_data - updated_means[i,:])).T, (train_data-updated_means[i,:]))
            covariances.append((1/points_per_cluster[i] * new_cov) + identity_cov_reg)
        return np.array(covariances)

    def _update_mixing_coefficients(self, responsibilities: FLOAT_ARR) -> FLOAT_ARR:
        """
        Helper function to update the mixing coefficients as part of the M step

        Args:
            responsibilities: responsibilities computed from the E-step

        Returns:
            Updated mixing coefficients
        """
        points_per_cluster = responsibilities.sum(axis=0)
        return points_per_cluster/responsibilities.sum()

    def _initialize_gmm(self, train_data: np.ndarray) -> Tuple[FLOAT_ARR, FLOAT_ARR, FLOAT_ARR]:
        """
        Initialize mean and mixing coefficients of size (N, C) where N is the
        dataset size and C is the number of clusters. This is done by splitting
        the dataset into C subsets and then calculating the mean and covariance
        for each subset. Note that this is done in-place.

        Args:
            train_data: Input training data
        Returns:
            Tuple of means, covariances, and mixing coefficients
        """
        if self._init:
            means, covariances, mixture_coeffs = self._initialize_w_kmeans(train_data)
        else:
            rng = np.random.default_rng()
            rng.shuffle(train_data)
            include = self.num_gaussians * (train_data.shape[0] // self.num_gaussians)
            
            splits = np.split(train_data[:include], self.num_gaussians)

            means = self._initialize_means(splits)
            covariances = self._initialize_covariances(splits)
            mixture_coeffs = self._initialize_mixing_coefficients()
        return means, covariances, mixture_coeffs

    def _initialize_w_kmeans(self, train_data: np.ndarray) -> Tuple[FLOAT_ARR, FLOAT_ARR, FLOAT_ARR]:
        kmeans = KMeans(self.num_gaussians)
        kmeans.fit(train_data)
        predictions = kmeans.predict(train_data)
        clusters = np.unique(predictions)
        identity_cov_reg = np.eye(train_data.shape[-1])*self.covariance_reg
        means, covariances, mixture_coeffs = [], [], []
        for c in clusters:
            cluster_data = train_data[predictions==c]
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_cov = np.cov(cluster_data.T) + identity_cov_reg
            cluster_coeff = len(cluster_data) / len(train_data)
            means.append(cluster_mean)
            covariances.append(cluster_cov)
            mixture_coeffs.append(cluster_coeff)

        means = np.array(means)
        covariances = np.array(covariances)
        mixture_coeffs = np.array(mixture_coeffs)
        return means, covariances, mixture_coeffs

    def _initialize_means(self, train_data_splits: List[FLOAT_ARR]) -> FLOAT_ARR:
        """
        Initialize the means 

        Args:
            train_data_splits: Input training data

        Returns:
            array of means
        """
        means = np.array([np.mean(split, axis=0) for split in train_data_splits])
        return means

    def _initialize_covariances(self, train_data_splits: List[FLOAT_ARR]) -> FLOAT_ARR:
        """
        Initialize the covariance matrices

        Args:
            train_data_splits: Input training data

        Returns:
            array of covariances
        """
        identity_cov_reg = np.eye(train_data_splits[0].shape[-1])*self.covariance_reg
        covariances = np.array([np.cov(split.T)+identity_cov_reg for split in train_data_splits])
        return covariances

    def _initialize_mixing_coefficients(self) -> FLOAT_ARR:
        """
        Initialize the mixing coefficients. The mixing coefficients should sum to 1

        Returns:
            array of mixing coefficients
        """
        rng = np.random.default_rng()
        mixing_coefficients = rng.random(size=(self.num_gaussians,))
        mixing_coefficients = np.array([i/mixing_coefficients.sum() for i in mixing_coefficients])
        return mixing_coefficients



def main():
    data, targets = load_digits(return_X_y=True)
    pca = PCA(0.99, whiten=True)
    data = pca.fit_transform(data)
    
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)
    num_clusters = len(np.unique(y_train))

    gmm = GMM(num_clusters, init="kmeans")
    gmm.fit(X_train)
    predictions_train = gmm.predict(X_train)
    print(predictions_train)
    train_accuracy = accuracy_score(y_train, predictions_train)
    print(f"Predictions on Train set: {train_accuracy}")
    predictions_test = gmm.predict(X_test)
    print(predictions_test)
    test_accuracy = accuracy_score(y_test, predictions_test)
    print(f"Predictions on Test set: {test_accuracy}")
    plt.plot(np.arange(len(gmm.log_likelihood_per_iter)), gmm.log_likelihood_per_iter)
    plt.savefig("log_likelihood.png")

if __name__ == '__main__':
    main()
