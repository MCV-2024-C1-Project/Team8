import abc
import numpy as np
from PIL import Image
from overrides import overrides

class HistogramDescriptor1D(abc.ABC):
    def __init__(self, bins: int = 256):
        self._bins = bins

    @abc.abstractmethod
    def preprocess_image(self, image: Image) -> np.array:
        """Given an PIL.Image object, apply any preprocessing before computing the histogram."""
        pass

    def compute(self, image: Image) -> np.array:
        processed_input_image = self.preprocess_image(image)
        assert processed_input_image.ndim == 2, "Image should be 2D (H, W)"
        histogram, _ = np.histogram(processed_input_image, bins=self._bins, range=(0, self._bins))
        return histogram

class GreyScaleHistogramDescriptor1D(HistogramDescriptor1D):
    def __init__(self, bins: int = 256):
        super().__init__(bins)
        self.color_space = "GreyScale"

    @overrides
    def preprocess_image(self, image: Image) -> np.array:
        return np.array(image.convert('L'))


class ColorHistogramDescriptor1D(HistogramDescriptor1D):
    def __init__(self, color_space: str, bins: int = 256):
        super().__init__(bins)
        self.color_space = color_space

    @overrides
    def preprocess_image(self, image: Image) -> np.array:
        return np.array(image.convert(self.color_space))

    @overrides
    def compute(self, image: Image) -> np.array:
        processed_input_image = self.preprocess_image(image)
        assert processed_input_image.ndim == 3, "Image should be (channels, H, W)"
        channels = np.split(processed_input_image, processed_input_image.shape[2], axis=2)
        #if self.color_space == 'LAB':
        #    channels = [channels[1], channels[2]]  # A and B channels
        histograms = [np.histogram(channel, bins=self._bins, range=(0, self._bins))[0] for channel in channels]
        return np.concatenate(histograms)

class MultiColorSpaceHistogramDescriptor1D():
    def __init__(self, color_spaces: list[str], bins: int = 256):
        self._bins = bins
        self.color_spaces = color_spaces
        self.color_space = ' - '.join(color_spaces)

    def preprocess_image(self, image: Image, color_space: str) -> np.array:
        return np.array(image.convert(color_space))

    def compute(self, image: Image) -> np.array:
        all_histograms = []

        for color_space in self.color_spaces:
            processed_input_image = self.preprocess_image(image, color_space)
            assert processed_input_image.ndim == 3, f"Image in color space {color_space} should be (H, W, C)"
            channels = np.split(processed_input_image, processed_input_image.shape[2], axis=2)
            histograms = [np.histogram(channel, bins=self._bins, range=(0, self._bins))[0] for channel in channels]
            all_histograms.append(np.concatenate(histograms))

        return np.concatenate(all_histograms)
