import numpy as np
from PIL.Image import Image


class DescriptorGenerator:

    def compute(self, input_image: Image, bins: int = 256) -> np.array:
        pass

class GreyScaleDescriptor(DescriptorGenerator):
    def compute(self, input_image: Image, bins: int = 256) -> np.array:
        histogram, _ = np.histogram(input_image.convert('L'), bins=bins, range=(0, bins - 1))
        return histogram
