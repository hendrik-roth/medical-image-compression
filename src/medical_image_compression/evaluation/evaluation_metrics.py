import os

from skimage.metrics import structural_similarity


class Evaluator:
    def __init__(self, compressed_image, reference_image,
                 compressed_image_path, reference_image_path, data_range=255):
        self.compressed_image_path = compressed_image_path
        self.reference_image_path = reference_image_path

        # calculate ssim based on Wang et al. 2004
        self.ssim = structural_similarity(reference_image, compressed_image,
                                          gaussian_weights=True,
                                          sigma=1.5,
                                          use_sample_covariance=False,
                                          data_range=data_range)
        # compression ratio
        self.cr = self.calc_cr()
        self.space_saving = 1 - self.cr

    def calc_cr(self):
        file_size_compressed = os.path.getsize(self.compressed_image_path)
        file_size_reference = os.path.getsize(self.reference_image_path)
        return file_size_reference / file_size_compressed
