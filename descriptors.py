# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:01:40 2024

@author: perel
"""
import abc
import numpy as np
from PIL import Image
from overrides import overrides
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
import pywt
import cv2
from skimage import feature

class TextureDescriptor:
    def __init__(self):
        pass
    
    # def name(self):
    #     return self.__class__.__name__

    def compute(self, image: np.array):
        raise NotImplementedError("This method should be overridden by subclasses")

    @staticmethod
    def to_grayscale(image: np.array):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

class LBPDescriptor(TextureDescriptor):
    def __init__(self, num_points=8, radius=1, color_space=cv2.COLOR_BGR2GRAY, num_bins = 128):  # Usually, num_points = 8*radius
        super().__init__()
        self.num_points = num_points
        self.radius = radius
        self.name = f"LBP_np_{num_points}_r_{radius}"
        self.color_space = color_space
        self.num_bins = num_bins

    def compute(self, image: np.array):
        image = cv2.cvtColor(image, self.color_space)

        lbp = local_binary_pattern(image, self.num_points, self.radius, method='uniform')

        hist, _ = np.histogram(lbp, bins=self.num_bins, range=(0, self.num_bins), density=True)
        #hist = hist.astype('float')
        #hist /= np.sum(hist) 
       
        return hist

class DCTDescriptor(TextureDescriptor):
    def __init__(self, N = 10, color_space=cv2.COLOR_BGR2HLS): 
        super().__init__()
        self.N = N
        self.name = f"DCT_{N}"
        self.color_space = color_space

    def zigzag_scan(self, matrix):
        rows, cols = matrix.shape
        zigzag = []
        for i in range(rows + cols - 1):
            if i % 2 == 0:
                for j in range(max(0, i - cols + 1), min(i + 1, rows)):
                    zigzag.append(matrix[j, i - j])
            else:
                for j in range(min(i, rows - 1), max(-1, i - cols), -1):
                    if 0 <= i - j < cols:  # Ensure the index is within bounds
                        zigzag.append(matrix[j, i - j])
        return zigzag

    def compute(self, image: np.array):
        image = cv2.cvtColor(image, self.color_space) if image.ndim == 3 else image

        descriptors = []
        for c in range(image.ndim): 
            dct_result = dct(dct(image[:, :, c], axis=0, norm='ortho'), axis=1, norm='ortho')
            desc = np.array(self.zigzag_scan(dct_result)[:self.N])
            descriptors.append(desc / (np.sum(desc)+1e-8))  # Normalize each channel's descriptor

        return np.concatenate(descriptors)


class WaveletDescriptor(TextureDescriptor):
    def __init__(self, wavelet='haar', level=5, color_space=cv2.COLOR_BGR2HLS, num_coeffs= 128): 
        super().__init__()
        self.wavelet = wavelet
        self.level = 4
        self.name = f"Wavelet_{wavelet}_lvl_{level}"
        self.color_space = color_space
        self.num_coeffs = num_coeffs

    def compute(self, image: np.array):
        image = cv2.cvtColor(image, self.color_space) if image.ndim == 3 else image
        descriptors = []
        for c in range(image.ndim):
            coeffs = pywt.wavedec2(image[:, :, c], wavelet=self.wavelet, level=self.level)[0].flatten()
            top_coeffs = np.sort(np.abs(coeffs))[-self.num_coeffs:] 

            descriptors.append(top_coeffs)
            
        return np.concatenate(descriptors)
    
class GLCMDescriptor(TextureDescriptor):
    def __init__(self, distances=[1], angles=[0],color_space= cv2.COLOR_BGR2GRAY, levels=256):
        self.name = "GLCM"
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.color_space = color_space

    def compute(self, image):
        image = cv2.cvtColor(image, self.color_space)
        if len(image.shape) != 2:
            raise ValueError("Input block must be a 2D grayscale image.")

        # Compute GLCM
        glcm = feature.greycomatrix(image, self.distances, self.angles, levels=self.levels, symmetric=True, normed=True)
        
        # Extract features
        features = []
        for i in range(len(self.distances)):
            contrast = feature.greycoprops(glcm, 'contrast')[i]
            dissimilarity = feature.greycoprops(glcm, 'dissimilarity')[i]
            homogeneity = feature.greycoprops(glcm, 'homogeneity')[i]
            energy = feature.greycoprops(glcm, 'energy')[i]
            correlation = feature.greycoprops(glcm, 'correlation')[i]
            features.extend([contrast, dissimilarity, homogeneity, energy, correlation])

        return np.array(features)
    
class GaborDescriptor(TextureDescriptor):
    def __init__(self, wavelengths=[3, 5, 7], orientations=[0, np.pi/4, np.pi/2,  3*np.pi/4], sigma=2,color_space= cv2.COLOR_BGR2GRAY):
        super().__init__()
        self.wavelengths = wavelengths
        self.orientations = orientations
        self.sigma = sigma
        self.name = f"Gabor_wavelengths_{wavelengths}_orientations_{len(orientations)}"
        self.color_space = color_space

    def compute(self, image: np.array):
        """Compute Gabor features for the given image block."""
        image = cv2.cvtColor(image, self.color_space)
        features = []

        for wavelength in self.wavelengths:
            for orientation in self.orientations:
                # Create Gabor kernel using OpenCV
                gabor_kernel = cv2.getGaborKernel(ksize=(31, 31), 
                                                   sigma=self.sigma, 
                                                   theta=orientation, 
                                                   lambd=wavelength, 
                                                   gamma=1, 
                                                   psi=0)

                # Convolve image with Gabor kernel
                filtered = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)

                # Compute the mean and standard deviation of the filtered image
                mean = np.mean(filtered)
                stddev = np.std(filtered)
                features.append(mean)
                features.append(stddev)

        return np.array(features)

print('1')