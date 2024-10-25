import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.signal import wiener
from PIL import Image

class RGBImageDenoiser:
    @staticmethod
    def split_channels(image):
        """Split RGB image into its three color channels."""
        return cv2.split(image)

    @staticmethod
    def merge_channels(channels):
        """Merge the RGB channels back together and clip to valid range [0, 255]."""
        merged_image = cv2.merge(channels)
        return np.clip(merged_image, 0, 255).astype(np.uint8)


class LinearFilters(RGBImageDenoiser):
    @staticmethod
    def gaussian_blur(image, kernel_size=(3, 3)):
        channels = RGBImageDenoiser.split_channels(image)
        denoised_channels = [cv2.GaussianBlur(channel, kernel_size, 0) for channel in channels]
        return RGBImageDenoiser.merge_channels(denoised_channels)

    @staticmethod
    def mean_filter(image, kernel_size=(3, 3)):
        channels = RGBImageDenoiser.split_channels(image)
        denoised_channels = [cv2.blur(channel, kernel_size) for channel in channels]
        return RGBImageDenoiser.merge_channels(denoised_channels)

    @staticmethod
    def wiener_filter(image, kernel_size=3):
        channels = RGBImageDenoiser.split_channels(image)
        denoised_channels = []

        for channel in channels:
            denoised_channel = wiener(channel, kernel_size)
            denoised_channel[np.isnan(denoised_channel)] = 0
            denoised_channel[denoised_channel < 0] = 0
            denoised_channels.append(denoised_channel)

        return RGBImageDenoiser.merge_channels(denoised_channels)


class NonLinearFilters(RGBImageDenoiser):
    @staticmethod
    def median_filter(image, kernel_size=3):
        channels = RGBImageDenoiser.split_channels(image)
        denoised_channels = [cv2.medianBlur(channel, kernel_size) for channel in channels]
        return RGBImageDenoiser.merge_channels(denoised_channels)

    @staticmethod
    def bilateral_filter(image, diameter=5, sigma_color=25, sigma_space=25):
        channels = RGBImageDenoiser.split_channels(image)
        denoised_channels = [
            cv2.bilateralFilter(channel, diameter, sigma_color, sigma_space)
            for channel in channels
        ]
        return RGBImageDenoiser.merge_channels(denoised_channels)

    @staticmethod
    def non_local_means_filter(image, h=0.6, patch_size=3, patch_distance=5, fast_mode=True):
        sigma_est = np.mean([
            estimate_sigma(image[..., channel], channel_axis=None)
            for channel in range(image.shape[-1])
        ])
        denoised_image = denoise_nl_means(
            image, h=h * sigma_est, patch_size=patch_size,
            patch_distance=patch_distance, channel_axis=-1, fast_mode=fast_mode
        )
        return RGBImageDenoiser.merge_channels([
            denoised_image[..., i] * 255 for i in range(denoised_image.shape[-1])
        ])


def denoise_image(image: Image, denoising_filter = NonLinearFilters.median_filter ) -> Image:
    img_np = np.array(image)
    result = denoising_filter(img_np)
    return Image.fromarray(result, 'RGB')
