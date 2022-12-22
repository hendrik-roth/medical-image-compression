import numpy as np
import torch
import cv2
import piq

from pydicom import dcmread
from skimage.io import imread
from skimage import exposure, img_as_ubyte

from skimage.metrics import structural_similarity


class Miqa:
    def __init__(self, reference_image, compressed_image, path_reference_img, path_compressed_img):
        self.fsim = 0
        self.reference_image = reference_image
        self.compressed_image = compressed_image
        self.path_reference = path_reference_img
        self.path_compressed = path_compressed_img

    def get_miqa_metrics(self):
        """calculating ssim and fsim

        Returns:
            tuple: (ssim, fsim)
        """
        dtype_ranges = {np.dtype('uint8'): 255, np.dtype("uint16"): 65535,
                        np.dtype("uint32"): (2 ** 32) - 1, float: 1, np.dtype('int8'): 128,
                        np.dtype('int16'): -32768, np.dtype('int32'): 2 ** 31}
        data_range = dtype_ranges[self.reference_image.pixel_array.dtype]
        _ssim = structural_similarity(self.reference_image.pixel_array,
                                      self.compressed_image.pixel_array,
                                      gaussian_weights=True,
                                      sigma=1.5,
                                      use_sample_covariance=False,
                                      data_range=data_range)
        self.calc_fsim()
        return _ssim, self.fsim

    @torch.no_grad()
    def calc_fsim(self):
        """calculate fsim metric

        based on Kastryulin et al. [1] GitHub implementation 
        https://github.com/photosynthesis-team/piq

        [1] S. Kastryulin, J. Zakirov, D. Prokopenko, and D. V. Dylov, 
        PyTorch Image Quality: Metrics for Image Quality Assessment: arXiv. [Online].
        Available: https://doi.org/10.48550/arxiv.2208.14818


        """
        # transform greyscale into rgb channels
        reference = imread(self.path_reference)
        reference_8bit = img_as_ubyte(exposure.rescale_intensity(reference))
        reference_rgb_channels = cv2.cvtColor(reference_8bit, cv2.COLOR_GRAY2RGB)

        compressed = dcmread(self.path_compressed).pixel_array
        compressed_8bit = img_as_ubyte(exposure.rescale_intensity(compressed))
        compressed_rgb_channels = cv2.cvtColor(compressed_8bit, cv2.COLOR_GRAY2RGB)

        # create image tensors
        x = torch.tensor(reference_rgb_channels).permute(2, 0, 1)[None, ...] / 255.
        y = torch.tensor(compressed_rgb_channels).permute(2, 0, 1)[None, ...] / 255.

        if torch.cuda.is_available():
            # Move to GPU to make computaions faster
            x = x.cuda()
            y = y.cuda()

        fsim_index: torch.Tensor = piq.fsim(x, y, data_range=255, reduction='none', chromatic=False)
        self.fsim = float(f"{fsim_index.item():0.4f}")
