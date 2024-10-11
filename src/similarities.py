import numpy as np
import abc
from overrides import overrides

class SimilarityMeasure(abc.ABC):
    @abc.abstractmethod
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        pass

class MSE(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Compute pairwise squared differences
        # Broadcast the query and database descriptors and compute the squared difference
        differences = query_descriptors[:, np.newaxis, ...] - database_descriptors[np.newaxis, :, ...]
        squared_diff = np.square(differences)

        # Mean over the last dimensions (histogram dimensions) to get MSE between each pair of points
        return squared_diff.mean(axis=tuple(range(2, squared_diff.ndim)))

class L1Distance(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Compute pairwise absolute differences
        differences = query_descriptors[:, np.newaxis, ...] - database_descriptors[np.newaxis, :, ...]
        return np.abs(differences).sum(axis=tuple(range(2, differences.ndim)))

class Bhattacharyya(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        epsilon = 1e-10  # Small constant to avoid division by zero

        # Normalize histograms across all histogram dimensions
        query_normalized = query_descriptors / (np.sum(query_descriptors, axis=tuple(range(1, query_descriptors.ndim)), keepdims=True) + epsilon)
        database_normalized = database_descriptors / (np.sum(database_descriptors, axis=tuple(range(1, database_descriptors.ndim)), keepdims=True) + epsilon)

        # Compute Bhattacharyya distance
        product = query_normalized[:, np.newaxis, ...] * database_normalized[np.newaxis, :, ...]

        # Sum over all histogram dimensions beyond the first two (query and database dimensions)
        # To ensure the shape is reduced to (30, 287), we need to sum over the remaining dimensions correctly
        sum_product = np.sum(np.sqrt(product), axis=tuple(range(2, product.ndim)))

        # Clip to avoid log(0)
        sum_product = np.clip(sum_product, epsilon, None)
        distances = -np.log(sum_product)

        # Return distances of shape (30, 287)
        return distances



class ChiSquaredDistance(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Compute pairwise squared differences
        differences = query_descriptors[:, np.newaxis, ...] - database_descriptors[np.newaxis, :, ...]
        sum_ = query_descriptors[:, np.newaxis, ...] + database_descriptors[np.newaxis, :, ...]

        # Add epsilon to prevent division by zero
        return 0.5 * np.sum((differences ** 2) / (sum_ + 1e-10), axis=tuple(range(2, differences.ndim)))


class HistogramIntersection(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Add epsilon to avoid division by zero during normalization
        eps = 1e-10

        # Normalize histograms by summing over all dimensions except the first (batch dimension)
        query_sums = np.sum(query_descriptors, axis=tuple(range(1, query_descriptors.ndim)), keepdims=True)
        query_normalized = query_descriptors / np.clip(query_sums, eps, None)

        database_sums = np.sum(database_descriptors, axis=tuple(range(1, database_descriptors.ndim)), keepdims=True)
        database_normalized = database_descriptors / np.clip(database_sums, eps, None)

        # Calculate intersection over the full histogram dimensions
        min_vals = np.minimum(query_normalized[:, np.newaxis, ...], database_normalized[np.newaxis, :, ...])

        # Sum over all histogram dimensions (starting from the 2nd axis onwards)
        intersection = np.sum(min_vals, axis=tuple(range(2, min_vals.ndim)))

        return -intersection

class HellingerKernel(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        epsilon = 1e-10  # Small constant to avoid division by zero

        # Normalize histograms across all histogram dimensions
        query_descriptors = query_descriptors / (np.sum(query_descriptors, axis=tuple(range(1, query_descriptors.ndim)), keepdims=True) + epsilon)
        database_descriptors = database_descriptors / (np.sum(database_descriptors, axis=tuple(range(1, database_descriptors.ndim)), keepdims=True) + epsilon)

        # Compute the square root of normalized histograms
        query_sqrt = np.sqrt(query_descriptors)
        database_sqrt = np.sqrt(database_descriptors)

        # Compute the pairwise differences
        differences = query_sqrt[:, np.newaxis, ...] - database_sqrt[np.newaxis, :, ...]

        # Compute the squared differences across all histogram dimensions
        squared_differences = np.sum(differences ** 2, axis=tuple(range(2, differences.ndim)))

        # Apply the Hellinger kernel formula
        return 1 - np.exp(-squared_differences / 2)

