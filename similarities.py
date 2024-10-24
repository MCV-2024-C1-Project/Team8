# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:21:36 2024

@author: perel
"""


import numpy as np
import abc
from overrides import overrides
from scipy.spatial import distance

class SimilarityMeasure(abc.ABC):
    @abc.abstractmethod
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        pass


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
    
class EuclideanDistance(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Compute the Euclidean distance between the query and database descriptors
        squeezed_array1 = np.squeeze(query_descriptors)  # Shape (30, 125)
        squeezed_array2 = np.squeeze(database_descriptors)
        dist_matrix = distance.cdist(squeezed_array1, squeezed_array2, metric='euclidean')
        return dist_matrix
    
class CosineSimilarity(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        squeezed_array1 = np.squeeze(query_descriptors)  
        squeezed_array2 = np.squeeze(database_descriptors)
        cosine_dist_matrix = distance.cdist(squeezed_array1, squeezed_array2, metric='cosine')
        return cosine_dist_matrix
    
class ManhattanDistance(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Squeeze arrays to remove single-dimensional entries
        squeezed_array1 = np.squeeze(query_descriptors)
        squeezed_array2 = np.squeeze(database_descriptors)
        # Compute Manhattan (L1) distance
        manhattan_dist_matrix = distance.cdist(squeezed_array1, squeezed_array2, metric='cityblock')
        return manhattan_dist_matrix
    
class KullbackLeiblerDivergence(SimilarityMeasure):
    @overrides
    def compute(self, query_descriptors: np.array, database_descriptors: np.array) -> np.array:
        # Normalize histograms to create probability distributions
        query_normalized = query_descriptors / np.sum(query_descriptors, axis=1, keepdims=True)
        database_normalized = database_descriptors / np.sum(database_descriptors, axis=1, keepdims=True)

        # To avoid division by zero and log(0), we add a small epsilon
        epsilon = 1e-10
        query_normalized = np.clip(query_normalized, epsilon, None)
        database_normalized = np.clip(database_normalized, epsilon, None)

        # Compute the KL divergence
        kl_divergence = np.sum(query_normalized[:, np.newaxis, :] * np.log(query_normalized[:, np.newaxis, :] / database_normalized[np.newaxis, :, :]), axis=2)

        return kl_divergence
    