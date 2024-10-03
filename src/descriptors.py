import numpy as np
from PIL.Image import Image


class DescriptorGenerator:
    def compute(self, input_image: Image, bins: int = 256) -> np.array:
        pass

class GreyScaleDescriptor(DescriptorGenerator):
    def compute(self, input_image: Image, bins: int = 256) -> np.array:
        histogram, _ = np.histogram(input_image.convert('L'), bins=bins, range=(0, bins))
        return histogram

# TODO: Combine RGB and HSL descriptors to reduce work for each image!

class RedDescriptor(DescriptorGenerator): 
    def compute(self, input_image: Image, bins: int = 256) -> np.ndarray:
        rgb_image = np.array(input_image)
        histogram, _ = np.histogram(rgb_image[:, :, 0], bins=bins, range=(0, 255))
        return histogram
    
class GreenDescriptor(DescriptorGenerator):
    def compute(self, input_image: Image, bins: int = 256) -> np.ndarray:
        rgb_image = np.array(input_image)
        histogram, _ = np.histogram(rgb_image[:, :, 1], bins=bins, range=(0, 255)) 
        return histogram
    
class BlueDescriptor(DescriptorGenerator):
    def compute(self, input_image: Image, bins: int = 256) -> np.ndarray:
        rgb_image = np.array(input_image)
        histogram, _ = np.histogram(rgb_image[:, :, 2], bins=bins, range=(0, 255))
        return histogram
    
# HSL

class HueDescriptor(DescriptorGenerator):
    def compute(self, input_image: Image, bins: int = 256) -> np.ndarray:
        hsv_image = input_image.convert("HSV")
        histogram, _ = np.histogram(np.array(hsv_image)[:, :, 0] , bins=bins, range=(0, 255))
        return histogram
    
class SaturationDescriptor(DescriptorGenerator):
    def compute(self, input_image: Image, bins: int = 256) -> np.ndarray:
        hsv_image = input_image.convert("HSV")
        histogram, _ = np.histogram(np.array(hsv_image)[:, :, 0] , bins=bins, range=(0, 255))
        return histogram
    
class ValueDescriptor(DescriptorGenerator):
    def compute(self, input_image: Image, bins: int = 256) -> np.ndarray:
        hsv_image = input_image.convert("HSV")
        histogram, _ = np.histogram(np.array(hsv_image)[:, :, 0] , bins=bins, range=(0, 255))
        return histogram

# TODO: Add CieLab, YCbCr, ...