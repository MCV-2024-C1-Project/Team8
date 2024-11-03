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
        """
        Initialize the SIFT descriptor with a limit on the number of keypoints.
        Args:
            max_features (int): Maximum number of keypoints to retain.
        """
        self.sift = cv2.SIFT_create()
        self.max_features = max_features

    def compute(self, image):
        """
        Compute SIFT descriptors for an image with filtering by response score.
        Args:
            image (np.ndarray): Input image.
        Returns:
            keypoints (list): List of filtered keypoints.
            descriptors (np.ndarray): Array of filtered SIFT descriptors.
        """
        # Detect all keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(image, None)

        # Sort keypoints by response and keep only the top N
        if len(keypoints) > self.max_features:
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:self.max_features]
            descriptors = descriptors[:self.max_features]

        return keypoints, descriptors

class ORBDescriptor:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def compute(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image is not a valid numpy array.")
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
                # Resize the patch to 64x128, standard size for HOG
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
        """
        Initialize the image retrieval system.
        Args:
            descriptors (dict): A dictionary of descriptor name to descriptor instance.
            descriptor_path (str): The path where descriptors will be saved.
        """
        self.descriptors = descriptors  # Dictionary like {'SIFT': SIFTDescriptor(), 'ORB': ORBDescriptor(), 'HOG': HOGDescriptor()}
        self.descriptor_path = Path(descriptor_path)
        self.descriptor_path.mkdir(parents=True, exist_ok=True)

    def compute_descriptors(self, image, descriptor_name):
        """
        Compute descriptors for a single image using the specified descriptor.
        Args:
            image (np.ndarray): The image to compute descriptors for.
            descriptor_name (str): The name of the descriptor to use (e.g., 'SIFT', 'ORB', 'HOG').
        Returns:
            np.ndarray: Computed descriptors for the image.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image is not a valid numpy array.")

        descriptor = self.descriptors.get(descriptor_name)
        if descriptor is None:
            raise ValueError(f"Descriptor {descriptor_name} not found.")

        if descriptor_name == 'HOG':
            # Compute SIFT keypoints and use them for HOG if required
            keypoints = self.descriptors['SIFT'].compute(image)[0]
            hog_descriptors = descriptor.compute(image, keypoints)
            return hog_descriptors if hog_descriptors is not None else np.array([])
        else:
            keypoints, descriptors = descriptor.compute(image)
            return descriptors if descriptors is not None else np.array([])

    def load_museum_descriptors(self, museum_images_np, descriptor_name):
        """
        Load or compute descriptors for museum images for a specific descriptor.
        Args:
            museum_images_np (list): List of museum images as numpy arrays.
            descriptor_name (str): The name of the descriptor to load or compute.
        Returns:
            list: List of descriptors for each museum image.
        """
        descriptor_file = self.descriptor_path / f"{descriptor_name}_descriptors.pkl"

        # Load from file if it exists and matches the number of images
        if descriptor_file.exists():
            print(f"Loading precomputed {descriptor_name} descriptors...")
            with open(descriptor_file, 'rb') as file:
                museum_descriptors = pickle.load(file)
            if len(museum_descriptors) == len(museum_images_np):
                return museum_descriptors
            else:
                print(
                    f"Descriptor file found, but not all descriptors are present. Recomputing missing descriptors for {descriptor_name}.")

        # Compute descriptors if they are missing or partially computed
        museum_descriptors = []
        for img in tqdm(museum_images_np, desc=f"Computing {descriptor_name} Descriptors", leave=False):
            museum_descriptors.append(self.compute_descriptors(img, descriptor_name))

        # Save the computed descriptors
        with open(descriptor_file, 'wb') as file:
            pickle.dump(museum_descriptors, file)

        return museum_descriptors

    def match_images(self, query_desc, museum_desc, descriptor_name):
        """
        Match query and museum descriptors using the specified descriptor and compute a score.
        Args:
            query_desc (np.ndarray): Query descriptors.
            museum_desc (np.ndarray): Museum descriptors.
            descriptor_name (str): The descriptor to use for matching.
        Returns:
            float: The matching score.
        """
        if query_desc.size == 0 or museum_desc.size == 0:
            return np.inf

        # Use vectorized distance calculation with cdist for efficiency
        metric = 'hamming' if descriptor_name == 'ORB' else 'euclidean'
        distances = cdist(query_desc, museum_desc, metric=metric)

        # Take the minimum distance for each query descriptor and compute the mean
        min_distances = np.min(distances, axis=1)
        return np.mean(min_distances)

    def retrieve_similar_images(self, query_images, museum_images, K, descriptor_name):
        """
        Retrieve the top K similar image indices for each sub-image in query images using an adaptive "elbow" method
        based on score gaps.

        Args:
            query_images (list): List of lists of query images as PIL Images.
            museum_images (list): List of museum images as PIL Images.
            K (int): Maximum number of similar images to retrieve.
            descriptor_name (str): The descriptor to use for matching (e.g., 'SIFT', 'ORB', 'HOG').

        Returns:
            list: A list where each element corresponds to a query image and is itself a list of sub-lists, where
                  each sub-list contains the top similar image indices for each sub-image in the query.
                  If no good match is found for a sub-image, the list contains `-1`.
            list: A list of the initial gaps between the first two scores for each sub-image.
        """
        # Convert museum images to numpy arrays once
        museum_images_np = [np.array(image) for image in museum_images]

        # Load or compute museum descriptors for the chosen descriptor
        museum_descriptors = self.load_museum_descriptors(museum_images_np, descriptor_name)
        print(f"Loaded {len(museum_descriptors)} museum descriptors.")

        gap_0 = []  # Store the first gaps for analysis
        results = []  # Store final results for each query image
        print(f"Processing query images with {descriptor_name}...")

        for query_set in tqdm(query_images, desc="Query Sets"):
            query_results = []  # Results for each sub-image in the current query image

            for query_image in query_set:
                # Convert and compute descriptors for the current sub-image
                query_image_np = np.array(query_image)
                query_desc = self.compute_descriptors(query_image_np, descriptor_name)
                scores = []

                # Calculate similarity scores for each museum image and store with index
                for museum_idx, museum_desc in enumerate(museum_descriptors):
                    score = self.match_images(query_desc, museum_desc, descriptor_name)
                    scores.append((museum_idx, score))  # Store (index, score)

                # Retrieve and sort top K matches by score in ascending order
                top_k_matches = nsmallest(K, scores, key=lambda x: x[1])
                top_k_indices = [index for index, score in top_k_matches]
                top_k_scores = [score for index, score in top_k_matches]

                # Calculate gaps between consecutive scores
                gaps = np.diff(top_k_scores)  # Consecutive differences between sorted scores
                if len(gaps) > 0:
                    gap_0.append([top_k_indices, gaps])  # Store the first gap for analysis

                # Determine the largest gap for potential cutoff
                if len(gaps) > 0:
                    largest_gap_index = np.argmax(gaps) + 1  # +1 to get the index in top_k_scores after the gap
                else:
                    largest_gap_index = 1  # If only one score, consider it a strong match

                # Define a dynamic threshold based on the gap distribution
                threshold = np.mean(gaps) + np.std(gaps) * 0.5

                # Determine the indices to keep based on the largest gap
                if largest_gap_index == 1 and gaps[0] > threshold:
                    # If the largest gap is after the first score and it's significantly larger than others
                    result_indices = top_k_indices  # Only the best match is returned
                else:
                    # Keep indices up to the position of the largest gap
                    result_indices = [-1]

                query_results.append(result_indices)  # Store results for the current sub-image

            results.append(query_results)  # Store results for the current query image (with multiple sub-images)

        return results, gap_0