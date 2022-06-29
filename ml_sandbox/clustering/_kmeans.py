from typing import Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml



FLOAT_ARR = npt.NDArray[np.float64]

def l2_distance(x: FLOAT_ARR, clusters: FLOAT_ARR) -> FLOAT_ARR:
    """
    Compute L2 distance

    Args:
        x: single data point
        clusters: the current cluster means

    Returns:
        L2 distances
    """
    return np.sqrt(np.square(x-clusters).sum(axis=1))

class KMeans():
    """
    KMeans Model implementation based on the algorithm presented in Section 9.1
    of Pattern Recogntion and Machine Learning by Christopher Bishop
    """
    def __init__(self, num_clusters: int, convergence_threshold: float=1e-6) -> None:
        self.num_clusters = num_clusters
        self.means = np.zeros(num_clusters)
        self.convergence_threshold = convergence_threshold
        self._cost_per_iter = []

    @property
    def cost_per_iter(self):
        return self._cost_per_iter

    def fit(self, train_data: FLOAT_ARR) -> None:
        """
        Train the model

        Args:
            train_data: Input training data
        """
        prev_cost = 1e6
        difference = 1e6
        count = 0

        self.means = self._initialize_means(train_data)
        while difference > self.convergence_threshold:
            distances, assignments = self._expectation_step(train_data)
            new_cost = self._cost(distances, assignments)
            difference = np.abs(new_cost - prev_cost)
            self.means = self._maximization_step(train_data, assignments)
            self._cost_per_iter.append(new_cost)
            prev_cost = new_cost
            print(f"=====Iteration {count}: Sum of Squares: {new_cost}, Diff:{difference}=====")
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
        _, predictions = self._expectation_step(test_data)
        return predictions


    def _expectation_step(self, train_data: FLOAT_ARR) -> Tuple[FLOAT_ARR, FLOAT_ARR]:
        """
        Evaluate the responsibilities using the current parameter values

        Args:
            train_data: input training data

        Returns:
            Tuple of distance matrix and cluster assignments
        """
        distances = self._get_l2_distances(train_data)
        assignments = self._assign_points(distances)
        return distances, assignments

    def _maximization_step(self, train_data: FLOAT_ARR, assignments: FLOAT_ARR) -> FLOAT_ARR:
        """
        Re-estimate the parameters using the current responsibilities

        Args:
            train_data: input training data
            assignments: cluster assignments

        Returns:
            Tuple of means, covariances, and mixture coefficients
        """
        means = self._update_means(train_data, assignments)
        return means

    def _cost(self, distances: FLOAT_ARR, assignments: FLOAT_ARR) -> FLOAT_ARR:
        """
        Compute the sum of square of the L2 distances

        Args:
            distances: distance vector that each point is from every cluster
            assignments: 

        Returns:
            sum of l2 distances
        """
        cost = np.sum(distances[np.arange(len(distances)), assignments])
        return cost

    def _update_means(self, train_data: FLOAT_ARR, assignments: FLOAT_ARR) -> FLOAT_ARR:
        """
        Helper function to update the means as part of the M step

        Args:
            train_data: input training data
            assignments: cluster assignments

        returns:
            Updated mean value
        """
        means = np.array(
            [train_data[assignments==i].sum(axis=0)/sum(assignments==i) for i in range(self.num_clusters)]
        )
        return means

    def _get_l2_distances(self, train_data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Get the distance matrix from each point to each cluster

        Args:
            train_data: input training data

        Returns:
            Distance matrix
        """
        return np.array([l2_distance(x, self.means) for x in train_data])

    def _assign_points(self, distances: FLOAT_ARR) -> FLOAT_ARR:
        """
        Assign points to their nearest clusters

        Args:
            distances: distance vector that each point is from every cluster

        Returns:
            cluster assignments
        """
        assignments = np.argmin(distances, axis=-1)
        return assignments

    def _initialize_means(self, train_data: FLOAT_ARR) -> FLOAT_ARR:
        """
        Initialize the means 

        Args:
            train_data: Input training data
            targets: Input target data

        Returns:
            array of means
        """
        include = self.num_clusters * (train_data.shape[0] // self.num_clusters)
        splits = np.split(train_data[:include], self.num_clusters)

        means = np.array([split.mean(axis=0) for split in splits])
        return means


def main():
    data, targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    ndims = int(np.sqrt(data.shape[-1]))
    
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)

    # Cheat, for now
    num_clusters = len(np.unique(y_train))

    kmeans = KMeans(num_clusters)
    kmeans.fit(X_train)
    predictions_train = kmeans.predict(X_train)
    train_accuracy = accuracy_score(y_train, predictions_train)
    predictions_test = kmeans.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions_test)

    print(f"Train Dataset - (Predictions, Truth):\n{*list(zip(predictions_train, y_train)),}")
    print(f"Test Dataset - (Predictions, Truth):\n{*list(zip(predictions_train, y_train)),}")
    print(f"Predictions on Train set: {train_accuracy}")
    print(f"Predictions on Test set: {test_accuracy}")

    for i, num in enumerate(kmeans.means):
        plt.imshow(num.reshape(ndims, ndims))
        plt.savefig(f"mean_{i}.png")
        plt.close()

    plt.plot(np.arange(len(kmeans.cost_per_iter)), np.array(kmeans.cost_per_iter))
    plt.savefig("costs.png")

if __name__ == '__main__':
    main()
