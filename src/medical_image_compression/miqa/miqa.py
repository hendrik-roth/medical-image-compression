import numpy as np

from skimage.metrics import structural_similarity
from image_similarity_measures.quality_metrics import fsim


class Miqa:
    def __init__(self, reference_image, compressed_image):
        self.reference_image = reference_image
        self.compressed_image = compressed_image

    def get_miqa_metrics(self):
        """calculating ssim and fsim

        Returns:
            tuple: (ssim, fsim)
        """
        dtype_ranges = {np.uint8: 255, np.uint16: 65535,
                        np.uint32: (2 ** 32) - 1, float: 1, np.int8: 128,
                        np.int16: -32768, np.int32: 2 ** 31}
        data_range = dtype_ranges[self.reference_image.pixel_array.dtype]
        _ssim = structural_similarity(self.reference_image.pixel_array,
                                      self.compressed_image.pixel_array,
                                      gaussian_weights=True,
                                      sigma=1.5,
                                      use_sample_covariance=False,
                                      data_range=data_range)

        _fsim = fsim(self.reference_image.pixel_array,
                     self.compressed_image.pixel_array)

        return _ssim, _fsim
