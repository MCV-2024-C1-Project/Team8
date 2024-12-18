import numpy as np
import abc
from overrides import overrides
from scipy.spatial import distance


class SimilarityMeasure(abc.ABC):
    @abc.abstractmethod
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        pass


class MSE(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Compute pairwise squared differences
        differences = query_descriptors[:, np.newaxis, :] - database_descriptors[np.newaxis, :, :]
        squared_diff = np.square(differences)

        # Mean over the last dimension (K) to get MSE between each pair of points
        return squared_diff.mean(axis=2)


class L1Distance(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        differences = query_descriptors[:, np.newaxis, :] - database_descriptors[np.newaxis, :, :]
        return np.abs(differences).sum(axis=2)


class Bhattacharyya(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Normalize histograms
        query_normalized = query_descriptors / np.sum(query_descriptors, axis=1, keepdims=True)
        database_normalized = database_descriptors / np.sum(database_descriptors, axis=1, keepdims=True)

        # This uses the fact that Bhattacharyya distance is defined as:
        distances = -np.log(
            np.sum(np.sqrt(query_normalized[:, np.newaxis, :] * database_normalized[np.newaxis, :, :]), axis=2))
        return distances


class ChiSquaredDistance(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        differences = query_descriptors[:, np.newaxis, :] - database_descriptors[np.newaxis, :, :]
        sum_ = query_descriptors[:, np.newaxis, :] + database_descriptors[np.newaxis, :, :]
        return 0.5 * np.sum((differences ** 2) / (sum_ + 1e-10), axis=2)


class HistogramIntersection(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Normalize histograms
        query_normalized = query_descriptors / np.sum(query_descriptors, axis=1, keepdims=True)
        database_normalized = database_descriptors / np.sum(database_descriptors, axis=1, keepdims=True)

        # Calculate intersection
        min_vals = np.minimum(query_normalized[:, np.newaxis, :], database_normalized[np.newaxis, :, :])
        intersection = np.sum(min_vals, axis=2)

        return -intersection


class HellingerKernel(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Normalize
        query_descriptors = query_descriptors / np.sum(query_descriptors, axis=1, keepdims=True)
        database_descriptors = database_descriptors / np.sum(database_descriptors, axis=1, keepdims=True)

        query_sqrt = np.sqrt(query_descriptors)
        database_sqrt = np.sqrt(database_descriptors)
        differences = query_sqrt[:, np.newaxis, :] - database_sqrt[np.newaxis, :, :]
        squared_differences = np.sum(differences ** 2, axis=2)
        return 1 - np.exp(-squared_differences / 2)
    
class CosineSimilarity(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        squeezed_array1 = np.squeeze(query_descriptors)  
        squeezed_array2 = np.squeeze(database_descriptors)
        cosine_dist_matrix = distance.cdist(squeezed_array1, squeezed_array2, metric='cosine')
        return cosine_dist_matrix
    