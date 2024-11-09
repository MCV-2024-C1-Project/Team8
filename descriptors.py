import abc
from PIL import Image
from overrides import overrides
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
import pywt
import cv2
from scipy.spatial.distance import cdist
from typing import List
from heapq import nsmallest
from tqdm import tqdm
import pickle
from pathlib import Path


###################################################################################################
############################ COLOR SPACE DESCRIPTORS ##############################################
###################################################################################################

class HistogramDescriptor(abc.ABC):
    def __init__(self, bins: int = 256, histogram_type: str = "default"):
        self._bins = bins
        self.histogram_type = histogram_type  # 'default', 'normalized', 'cumulative', 'log-chromatic'

    @property
    def name(self):
        return f"{self.__class__.__name__}_bins_{self._bins}_hist_type_{self.histogram_type}"

    @abc.abstractmethod
    def preprocess_image(self, image: Image) -> np.array:
        """Given an PIL.Image object, apply any preprocessing before computing the histogram."""
        pass

    def compute_histogram(self, image: np.array) -> np.array:
        """Compute histogram based on the selected type."""
        if self.histogram_type == 'default':
            hist, _ = np.histogram(image, bins=self._bins, range=(0, self._bins))
        elif self.histogram_type == 'normalized':
            hist, _ = np.histogram(image, bins=self._bins, range=(0, self._bins))
            hist = hist / hist.sum()  # Normalize the histogram
        elif self.histogram_type == 'cumulative':
            hist, _ = np.histogram(image, bins=self._bins, range=(0, self._bins))
            hist = np.cumsum(hist)  # Compute cumulative histogram
        elif self.histogram_type == 'log-chromatic':
            # Compute log-chromaticity histogram
            epsilon = 1e-5
            channel = image.astype(np.float32) + epsilon  # Avoid division by zero
            log_channel = np.log(channel / np.max(channel))  # Log chromaticity
            hist, _ = np.histogram(log_channel, bins=self._bins, range=(-5, 0))
        else:
            raise ValueError(f"Unsupported histogram type: {self.histogram_type}")

        return hist

    def compute(self, image: Image) -> np.array:
        processed_input_image = self.preprocess_image(image)
        assert processed_input_image.ndim == 2, "Image should be 2D (H, W)"
        histogram = self.compute_histogram(processed_input_image)
        return histogram

class GreyScaleHistogramDescriptor1D(HistogramDescriptor):
    def __init__(self, bins: int = 256, histogram_type: str = "default"):
        super().__init__(bins, histogram_type)

    @overrides
    def preprocess_image(self, image: Image) -> np.array:
        return np.array(image.convert('L'))


class ColorHistogramDescriptor1D(HistogramDescriptor):
    def __init__(self, color_space: str, bins: int = 256, histogram_type: str = "default"):
        super().__init__(bins, histogram_type)
        self.color_space = color_space

    @property
    @overrides
    def name(self):
        return f"{super().name}_{self.color_space}"

    @overrides
    def preprocess_image(self, image: Image) -> np.array:
        return np.array(image.convert(self.color_space))

    @overrides
    def compute(self, image: Image) -> np.array:
        processed_input_image = self.preprocess_image(image)
        assert processed_input_image.ndim == 3, "Image should be (channels, H, W)"
        channels = np.split(processed_input_image, processed_input_image.shape[2], axis=2)

        histograms = [np.histogram(channel, bins=self._bins, range=(0, self._bins))[0] for channel in channels]
        return np.concatenate(histograms)

class ColorHistogramDescriptor3D(ColorHistogramDescriptor1D):
    def __init__(self, color_space: str, bins: int = 256, histogram_type: str = "default"):
        super().__init__(color_space, bins, histogram_type)

    @overrides
    def compute(self, image: Image) -> np.array:
        processed_input_image = self.preprocess_image(image)
        assert processed_input_image.ndim == 3, "Image should be (channels, H, W)"

        hist_3d, _ = np.histogramdd(
            processed_input_image.reshape(-1, 3),
            bins=self._bins,
            range=((0, 255), (0, 255), (0, 255))
        )

        return hist_3d.ravel()

class MultiColorSpaceHistogramDescriptor1D(ColorHistogramDescriptor1D):
    def __init__(self, color_spaces: list[str], bins: int = 256, histogram_type: str = "default"):
        super().__init__('-'.join(color_spaces), bins, histogram_type)
        self.color_spaces = color_spaces

    @overrides(check_signature=False)
    def preprocess_image(self, image: Image, color_space: str) -> np.array:
        return np.array(image.convert(color_space))

    @overrides
    def compute(self, image: Image) -> np.array:
        all_histograms = []

        for color_space in self.color_spaces:
            processed_input_image = self.preprocess_image(image, color_space)
            assert processed_input_image.ndim == 3, f"Image in color space {color_space} should be (H, W, C)"
            channels = np.split(processed_input_image, processed_input_image.shape[2], axis=2)

            histograms = [self.compute_histogram(channel.squeeze()) for channel in channels]
            all_histograms.append(np.concatenate(histograms))

        return np.concatenate(all_histograms)


class MultiColorSpaceHistogramDescriptor3D(MultiColorSpaceHistogramDescriptor1D):
    def __init__(self, color_spaces: list[str], bins: int = 256, histogram_type: str = "default"):
        super().__init__(color_spaces, bins, histogram_type)

    @overrides
    def compute(self, image: Image) -> np.array:
        all_histograms = []

        for color_space in self.color_spaces:
            processed_input_image = self.preprocess_image(image, color_space)
            assert processed_input_image.ndim == 3, f"Image in color space {color_space} should be (H, W, C)"

            hist_3d, _ = np.histogramdd(
                processed_input_image.reshape(-1, 3),
                bins=self._bins,
                range=((0, 255), (0, 255), (0, 255))
            )

            all_histograms.append(hist_3d.ravel())

        return np.concatenate(all_histograms)

###################################################################################################
############################### TEXTURE DESCRIPTORS ###############################################
###################################################################################################

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
    def __init__(self, num_points=8, radius=1, color_space=cv2.COLOR_BGR2HLS):  # Usually, num_points = 8*radius
        super().__init__()
        self.num_points = num_points
        self.radius = radius
        self.name = f"LBP_np_{num_points}_r_{radius}"
        self.color_space = color_space

    def compute(self, image: np.array):
        image = cv2.cvtColor(image, self.color_space) if image.ndim == 3 else image

        descriptors = []
        for c in range(image.ndim):
            lbp = local_binary_pattern(image[:, :, c], self.num_points, self.radius, method='uniform').ravel()
            hist = np.histogram(lbp.ravel(), bins=np.arange(0, self.num_points + 3), range=(0, self.num_points + 2))[0].astype(float)
            hist /= (np.sum(hist.astype(float)) + 1e-8)  
            descriptors.append(hist / (np.sum(hist)+1e-8))  
        return np.concatenate(descriptors)

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
    def __init__(self, wavelet='haar', N = 10, color_space=cv2.COLOR_BGR2HLS): 
        super().__init__()
        self.wavelet = wavelet
        self.N = N
        self.name = f"Wavelet_{wavelet}_N_{self.N}"
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
            coeffs = pywt.wavedec2(image[:,:,c], wavelet='haar')
            arr, _ = pywt.coeffs_to_array(coeffs)
            # arr = (arr - arr.min()) / (arr.max() - arr.min()+1e-8)
            descriptors.append(np.array(self.zigzag_scan(arr)[:self.N]))
        return np.concatenate(descriptors)

    
    
class GaborDescriptor(TextureDescriptor):
    def __init__(self, wavelengths=[3, 5, 7], orientations=[0, np.pi/4, np.pi/2,  3*np.pi/4], sigma=2,color_space= cv2.COLOR_BGR2GRAY):
        super().__init__()
        self.wavelengths = wavelengths
        self.orientations = orientations
        self.sigma = sigma
        self.name = f"Gabor_wavelengths_{wavelengths}_orientations_{len(orientations)}".replace('[', '(').replace(']', ')')
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

###################################################################################################
############################### KEYPOINT DESCRIPTORS ##############################################
###################################################################################################

class SIFTDescriptor:
    def __init__(self, max_features=500):
        self.sift = cv2.SIFT_create() # arguments !
        self.max_features = max_features

    def compute(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if len(keypoints) > self.max_features:
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:self.max_features]
            descriptors = descriptors[:self.max_features]
        return keypoints, descriptors


class ORBDescriptor:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def compute(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image is not valid.")
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors


class HOGDescriptor:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()

    def compute(self, image, keypoints):
        hog_descriptors = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            patch = self._extract_patch(image, x, y)
            if patch is not None:
                patch_resized = cv2.resize(patch, (64, 128))
                hog_descriptor = self.hog.compute(patch_resized)
                hog_descriptors.append(hog_descriptor.flatten())
        return np.array(hog_descriptors)

    def _extract_patch(self, image, x, y, size=64):
        half_size = size // 2
        if x - half_size < 0 or y - half_size < 0 or x + half_size > image.shape[1] or y + half_size > image.shape[0]:
            return None
        return image[y-half_size:y+half_size, x-half_size:x+half_size]


class ImageRetrievalSystem:
    def __init__(self, descriptors, descriptor_path):
        self.descriptors = descriptors
        self.descriptor_path = Path(descriptor_path)
        self.descriptor_path.mkdir(parents=True, exist_ok=True)

    def compute_descriptors(self, image, descriptor_name):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image is not valid.")
        
        descriptor = self.descriptors.get(descriptor_name)
        if descriptor is None:
            raise ValueError(f"Descriptor {descriptor_name} not found.")

        # What if there is a HOG desc. and no SIFT ?
        if descriptor_name == 'HOG':
            keypoints = self.descriptors['SIFT'].compute(image)[0]
            return descriptor.compute(image, keypoints) if keypoints else np.array([])
        else:
            return descriptor.compute(image)[1] if descriptor.compute(image)[1] is not None else np.array([])

    def load_museum_descriptors(self, museum_images_np, descriptor_name):
        descriptor_file = self.descriptor_path / f"{descriptor_name}_descriptors.pkl"

        if descriptor_file.exists():
            with open(descriptor_file, 'rb') as file:
                museum_descriptors = pickle.load(file)
            if len(museum_descriptors) == len(museum_images_np):
                return museum_descriptors

        museum_descriptors = [self.compute_descriptors(img, descriptor_name) for img in tqdm(museum_images_np, desc=f"Computing {descriptor_name} Descriptors", leave=False)]
        with open(descriptor_file, 'wb') as file:
            pickle.dump(museum_descriptors, file)
        return museum_descriptors

    def match_images(self, query_desc, museum_desc, descriptor_name):
        if query_desc.size == 0 or museum_desc.size == 0:
            return np.inf

        metric = 'hamming' if descriptor_name == 'ORB' else 'euclidean'
        distances = cdist(query_desc, museum_desc, metric=metric)
        return np.mean(np.min(distances, axis=1))
    
    def count_good_matches(self, query_descriptors, museum_descriptors, distance_threshold=50):
        # Initialize list to hold count of good matches for each db image
        good_match_counts = []
    
        # BFMatcher for Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        # Loop over each database image's descriptors
        for museum_desc in museum_descriptors:
            
         
            if museum_desc.shape == (0,):
                match_count = 0
            
            else:
                # Find matches between the query and the current db image
                matches = bf.match(query_descriptors, museum_desc)
            
                # Filter matches based on distance threshold to count only "good matches"
                good_matches = [m for m in matches if m.distance < distance_threshold]
                match_count = len(good_matches)
            
            # Append the count of good matches for this db image to the list
            good_match_counts.append(match_count)
        
        return good_match_counts

    def retrieve_similar_images(self, query_images, museum_images, K, descriptor_name, t=0.995):

        museum_images_np = [np.array(image) for image in museum_images]
        #museum_images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in museum_images_np]
        museum_descriptors = self.load_museum_descriptors(museum_images_np, descriptor_name)
        check_gm = []
        #query_images = [np.array(image) for image in query_images]
        #query_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in query_images]
        print('Aqui')
        results, gap_0 = [], []
        i = 0
        for query_set in tqdm(query_images, desc="Query Sets"):
            query_results = []
            query_matches = []
            for query_image in query_set:
                query_image = np.array(query_image)
                #query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
                query_desc = self.compute_descriptors(query_image, descriptor_name)
                good_match_counts = self.count_good_matches(query_desc, museum_descriptors, distance_threshold=30)

                if max(good_match_counts) < 5:
                # No db image has enough good matches, so we set result to -1
                    result_indices = [-1]
                
                else:
                    scores = [(museum_idx, self.match_images(query_desc, museum_desc, descriptor_name)) 
                            for museum_idx, museum_desc in enumerate(museum_descriptors)]
                    top_k_matches = nsmallest(K, scores, key=lambda x: x[1])
                    top_k_indices = [index for index, score in top_k_matches]
                    top_k_scores = [score for index, score in top_k_matches]
                    
                    top_2_matches =  nsmallest(2, scores, key=lambda x: x[1])
                    top_2_scores = [score for index, score in top_2_matches]
    
                    gaps = np.diff(top_k_scores)
                    gap_0.append([top_k_indices, gaps])
    
                    first_distance = top_2_scores[0]
                    second_distance = top_2_scores[1]
                    result_indices = top_k_indices

                # print(i, first_distance / second_distance, top_k_indices, top_k_scores, sep='\n')
                '''
                # This method looks like the best for k = 1, implement another method for k>1 ?
                if first_distance < t* second_distance:
                    result_indices = top_k_indices
                else:
                    result_indices = [-1]
                '''
                query_results.append(result_indices)
                query_matches.append(good_match_counts)
                i += 1

            results.append(query_results)
            check_gm.append(query_matches)

        return results, gap_0, check_gm

print('1')