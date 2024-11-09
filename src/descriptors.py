import abc
from PIL import Image
from overrides import overrides
import numpy as np
from skimage.feature import local_binary_pattern, hog
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

class SIFTDescriptor: # Scale Invariant Feature Transform
    def __init__(self, max_features=500):
        self.sift = cv2.SIFT_create()
        self.max_features = max_features

    def compute(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        '''if len(keypoints) > self.max_features:
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:self.max_features]
            descriptors = descriptors[:self.max_features]'''
        return keypoints, descriptors


class ORBDescriptor: # (FAST + BRIEF)...
    def __init__(self):
        self.orb = cv2.ORB_create()

    def compute(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image is not valid.")
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

class HOGDescriptor:
    def __init__(self, pixels_per_cell=8, cells_per_block=2, win_size=256):
        self.pixels_per_cell = (pixels_per_cell, pixels_per_cell)
        self.cells_per_block = (cells_per_block, cells_per_block)
        self.win_size = (win_size, win_size)

    def compute(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(image, (256, 256))

        descriptors = hog(
            resized_image,
            orientations=9,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        
        return descriptors.astype(np.float32)


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
        
        descriptors = descriptor.compute(image)
        if len(descriptors)==2:
            keypoints, descriptors = descriptors

        return descriptors

    def load_museum_descriptors(self, museum_images, descriptor_name):
        descriptor_file = self.descriptor_path / f"{descriptor_name}_descriptors.pkl"

        if descriptor_file.exists():
            with open(descriptor_file, 'rb') as file:
                museum_descriptors = pickle.load(file)
            if len(museum_descriptors) == len(museum_images):
                return museum_descriptors

        museum_descriptors = [self.compute_descriptors(img, descriptor_name) for img in tqdm(museum_images, desc=f"Computing {descriptor_name} Descriptors", leave=False)]
        with open(descriptor_file, 'wb') as file:
            pickle.dump(museum_descriptors, file)
        return museum_descriptors

    def _get_flann(self):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        return flann


    def retrieve_similar_images(self, descriptor_name, museum_images, query_images, gt_list = None, K=1):
        results = []
        flann = self._get_flann()

        if gt_list == None:
            gt_list = [-2] * len(query_images)

        museum_descriptors = self.load_museum_descriptors(museum_images, descriptor_name)

        for idx, (query_images, gt_tuple) in enumerate(zip(query_images, gt_list)):
            print(f"Query {idx}")
            query_result = []
            for img_idx, image in enumerate(query_images):
                print(f" - Image {img_idx}")
                query_descriptor = self.compute_descriptors(np.array(image), descriptor_name)

                threshold = 0.03 if descriptor_name[:3]=='HOG' else 0.2
                metric = 'knn' if descriptor_name[:4]=='SIFT' else 'euclidean'

                img_results = self._get_img_results(query_descriptor, museum_descriptors, flann, metric)
                best_candidate = self._determine_best_candidate(img_results, query_descriptor, museum_descriptors, flann, K, metric, threshold)
                # print(best_candidate)
                if best_candidate[0][1] < 0.1:
                    best_candidate = [(-1, 1)]
                query_result.append(best_candidate)

                # Print and compare with ground truth
                print(f" Results : {query_result[img_idx][0]}")
                if gt_list[0] != -2:
                    print(f" GT : {gt_tuple[img_idx]}")
                    if query_result[img_idx][0][0] != gt_tuple[img_idx]:
                        print("Mismatch detected")
                print("############\n\n")
            results.append(query_result)

        return results

    def _compute_similarity(self, descriptor1, descriptor2, flann=None, metric='euclidean'):
        if metric=='euclidean':
            similarity = 1 - np.linalg.norm(descriptor1 - descriptor2) / np.linalg.norm(descriptor1 + descriptor2)
            return similarity
        if metric=='knn':
            matches = flann.knnMatch(descriptor1, descriptor2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            similarity_score = len(good) / len(matches) if matches else 0
            return similarity_score
            
    def _get_img_results(self, query_descriptor, museum_descriptors, flann, reverse=False, metric='euclidean'):
        img_results = []
        for museum_idx, db_descriptor in tqdm(enumerate(museum_descriptors)):
            if db_descriptor is None:
                continue
            if reverse:
                similarity_score = self._compute_similarity(db_descriptor, query_descriptor, flann, metric)
            else:
                similarity_score = self._compute_similarity(query_descriptor, db_descriptor, flann, metric)
            
            img_results.append((museum_idx, similarity_score))
        img_results.sort(key=lambda x: x[1], reverse=True)
        return img_results


    def _determine_best_candidate(self, img_results, query_descriptor, museum_descriptors, flann=None, K=1, metric='euclidean', threshold = 0.2):
        if len(img_results) > 1:
            top_score = img_results[0][1]
            second_score = img_results[1][1]
            relative_gap = (top_score - second_score) / top_score if top_score != 0 else 0
            print("   - Relative gap: ", relative_gap)
            ambiguous = relative_gap < threshold
        else:
            ambiguous = False

        if ambiguous:
            print("Ambiguous result detected")
            reverse_results = self._get_img_results(query_descriptor, museum_descriptors, flann, True, metric)
            reverse_top_score = reverse_results[0][1]
            if (reverse_top_score - second_score) / reverse_top_score >= threshold:
                if reverse_top_score > top_score:
                    best_candidate = reverse_results[:K]
                else:
                    best_candidate = img_results[:K]
            else:
                print("Detected as not found")
                best_candidate = [(-1, 1)]  # Mark as ambiguous
        else:
            best_candidate = img_results[:K] if img_results else [(-1, 1)]
        return best_candidate

