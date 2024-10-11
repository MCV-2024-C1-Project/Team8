import abc
import numpy as np
from PIL import Image
from overrides import overrides

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

class ColorHistogramDescriptor3D(HistogramDescriptor):
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

        hist_3d, edges = np.histogramdd(
            processed_input_image.reshape(-1, 3),
            bins=self._bins,
            range=((0, 255), (0, 255), (0, 255))
        )

        return hist_3d


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
