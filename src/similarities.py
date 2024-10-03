import numpy as np


class SimilarityMeasure:
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        pass

class MSE(SimilarityMeasure):
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Compute pairwise squared differences
        differences = query_descriptors[:, np.newaxis, :] - database_descriptors[np.newaxis, :, :]
        squared_diff = np.square(differences)

        # Mean over the last dimension (K) to get MSE between each pair of points
        return squared_diff.mean(axis=2)

class L1Distance(SimilarityMeasure):
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        differences = query_descriptors[:, np.newaxis, :] - database_descriptors[np.newaxis, :, :]
        return np.abs(differences).sum(axis=2)

class ChiSquaredDistance(SimilarityMeasure):
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        differences = query_descriptors[:, np.newaxis, :] - database_descriptors[np.newaxis, :, :]
        sum_ = query_descriptors[:, np.newaxis, :] + database_descriptors[np.newaxis, :, :]
        return 0.5 * np.sum((differences ** 2) / (sum_ + 1e-10), axis=2)

class HistogramIntersection(SimilarityMeasure):
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        min_vals = np.minimum(query_descriptors[:, np.newaxis, :], database_descriptors[np.newaxis, :, :])
        return np.sum(min_vals, axis=2)

class HellingerKernel(SimilarityMeasure):
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Normalize
        query_descriptors = query_descriptors / np.sum(query_descriptors, axis=1, keepdims=True)
        database_descriptors = database_descriptors / np.sum(database_descriptors, axis=1, keepdims=True)

        query_sqrt = np.sqrt(query_descriptors)
        database_sqrt = np.sqrt(database_descriptors)
        differences = query_sqrt[:, np.newaxis, :] - database_sqrt[np.newaxis, :, :]
        squared_differences = np.sum(differences ** 2, axis=2)
        return np.exp(-squared_differences / 2)