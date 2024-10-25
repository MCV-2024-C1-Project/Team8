import abc
import numpy as np
from PIL import Image
from overrides import overrides
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
import pywt
import cv2

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
    def __init__(self, wavelet='haar', level=5, color_space=cv2.COLOR_BGR2HLS): 
        super().__init__()
        self.wavelet = wavelet
        self.level = 4
        self.name = f"Wavelet_{wavelet}_lvl_{level}"
        self.color_space = color_space

    def compute(self, image: np.array):
        image = cv2.cvtColor(image, self.color_space) if image.ndim == 3 else image
        descriptors = []
        for c in range(image.ndim):
            coeffs = pywt.wavedec2(image[:, :, c], wavelet=self.wavelet, level=self.level)[0].flatten()

            hist = np.histogram(coeffs, bins=256, range=(0, np.max(coeffs) + 1e-8))[0].astype(float)
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize histogram

            descriptors.append(hist / (np.sum(hist)+1e-8)) # norm

        return np.concatenate(descriptors)
    
    
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

