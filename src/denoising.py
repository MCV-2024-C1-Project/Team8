import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.signal import wiener
from PIL import Image

class ColorSpaceImageDenoiser:
    def __init__(self, color_space='RGB'):
        self.color_space = color_space

    def to_color_space(self, image):
        """Convert RGB image to specified color space."""
        if self.color_space == 'YCbCr':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif self.color_space == 'Lab':
            return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        elif self.color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return image  # RGB (no conversion)

    def from_color_space(self, image):
        """Convert image from specified color space back to RGB."""
        if self.color_space == 'YCbCr':
            return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
        elif self.color_space == 'Lab':
            return cv2.cvtColor(image, cv2.COLOR_Lab2RGB)
        elif self.color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image  # RGB (no conversion)

    @staticmethod
    def split_channels(image):
        """Split image into its color channels."""
        return cv2.split(image)

    @staticmethod
    def merge_channels(channels):
        """Merge channels back together and clip to valid range [0, 255]."""
        merged_image = cv2.merge(channels)
        return np.clip(merged_image, 0, 255).astype(np.uint8)


class LinearFilters(ColorSpaceImageDenoiser):
    def gaussian_blur(self, image, kernel_size=(3, 3)):
        image_cs = self.to_color_space(image)
        channels = self.split_channels(image_cs)
        denoised_channels = [cv2.GaussianBlur(channel, kernel_size, 0) for channel in channels]
        return self.from_color_space(self.merge_channels(denoised_channels))

    def mean_filter(self, image, kernel_size=(3, 3)):
        image_cs = self.to_color_space(image)
        channels = self.split_channels(image_cs)
        denoised_channels = [cv2.blur(channel, kernel_size) for channel in channels]
        return self.from_color_space(self.merge_channels(denoised_channels))

    def wiener_filter(self, image, kernel_size=3):
        image_cs = self.to_color_space(image)
        channels = self.split_channels(image_cs)
        denoised_channels = []

        for channel in channels:
            denoised_channel = wiener(channel, kernel_size)
            denoised_channel[np.isnan(denoised_channel)] = 0
            denoised_channel[denoised_channel < 0] = 0
            denoised_channels.append(denoised_channel)

        return self.from_color_space(self.merge_channels(denoised_channels))


class NonLinearFilters(ColorSpaceImageDenoiser):
    def median_filter(self, image, kernel_size=3):
        image_cs = self.to_color_space(image)
        channels = self.split_channels(image_cs)
        denoised_channels = [cv2.medianBlur(channel, kernel_size) for channel in channels]
        return self.from_color_space(self.merge_channels(denoised_channels))

    def bilateral_filter(self, image, diameter=5, sigma_color=25, sigma_space=25):
        image_cs = self.to_color_space(image)
        channels = self.split_channels(image_cs)
        denoised_channels = [
            cv2.bilateralFilter(channel, diameter, sigma_color, sigma_space)
            for channel in channels
        ]
        return self.from_color_space(self.merge_channels(denoised_channels))

    def non_local_means_filter(self, image, h=0.6, patch_size=3, patch_distance=5, fast_mode=True):
        image_cs = self.to_color_space(image)
        sigma_est = np.mean([
            estimate_sigma(image_cs[..., channel], channel_axis=None)
            for channel in range(image_cs.shape[-1])
        ])
        denoised_image = denoise_nl_means(
            image_cs, h=h * sigma_est, patch_size=patch_size,
            patch_distance=patch_distance, channel_axis=-1, fast_mode=fast_mode
        )
        return self.from_color_space(self.merge_channels([
            denoised_image[..., i] * 255 for i in range(denoised_image.shape[-1])
        ]))


def denoise_image(image: Image, denoising_filter, color_space='RGB') -> Image:
    denoiser = NonLinearFilters(color_space=color_space) if denoising_filter in NonLinearFilters.__dict__.values() else LinearFilters(color_space=color_space)
    img_np = np.array(image)
    result = denoising_filter(denoiser, img_np)  # Pass denoiser instance as self
    return Image.fromarray(result, 'RGB')
