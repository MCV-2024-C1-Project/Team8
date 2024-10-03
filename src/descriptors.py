import abc

import numpy as np
from PIL.Image import Image
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
        assert len(processed_input_image.shape) == 2, "Expected input image to be of size (H, W)"
        histogram, _ = np.histogram(processed_input_image, bins=self._bins, range=(0, self._bins))
        return histogram


class GreyScaleHistogramDescriptor1D(HistogramDescriptor1D):
    @overrides
    def preprocess_image(self, image: Image) -> np.array:
        return np.array(image.convert('L'))


class RGBHistogramDescriptor1D(HistogramDescriptor1D):
    @overrides
    def compute(self, image: Image) -> np.array:
        rgb_histograms = []
        red_channel, green_channel, blue_channel = image.split()
        rgb_histograms.append(super().compute(red_channel))
        rgb_histograms.append(super().compute(green_channel))
        rgb_histograms.append(super().compute(blue_channel))
        return np.concatenate(rgb_histograms)

    @overrides
    def preprocess_image(self, image: Image) -> np.array:
        return np.array(image)


class HSVHistogramDescriptor1D(HistogramDescriptor1D):
    @overrides
    def compute(self, image: Image) -> np.array:
        image = image.convert("HSV")

        hsv_histograms = []
        hue_channel, saturation_channel, value_channel = image.split()
        hsv_histograms.append(super().compute(hue_channel))
        hsv_histograms.append(super().compute(saturation_channel))
        hsv_histograms.append(super().compute(value_channel))
        return np.concatenate(hsv_histograms)

    @overrides
    def preprocess_image(self, image: Image) -> np.array:
        return np.array(image)

# TODO: Add CieLab, YCbCr, ...