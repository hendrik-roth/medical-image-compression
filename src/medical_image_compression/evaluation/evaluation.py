import pandas as pd

from pathlib import Path
from pydicom import dcmread
from time import time

from ..image_processing import ImageReader
from ..miqa import Miqa


class Evaluator:
    def __init__(self,
                 images,
                 compression_method,
                 out_path):
        """init evaluator

        Args:
            images (list): list with paths of dirs where the dicom images are stored
            compression_method (method): compression method used for compression
            out_path (str): path where the compressed new images should be stored
        """
        self.images = images
        self.compression_method = compression_method
        self.out_path = out_path

    def evaluate(self):
        """main function to perform complete evaluation

        Returns:
            pd.DataFrame: DataFrame with all evaluation metrics
        """
        metrics = self.compress_all_images()
        return self.final_sample_assessment(metrics)

    def compress_all_images(self):
        """peform activity process for all samples in set

        This method will do Step 1 - 4 described in the framework paper

        Returns:
            dict: Dictionary with all metrics measured as lists. Keys are the metrics.
        """
        t_list, cr_list, ssim_list, fsim_list = ([] for _ in range(4))

        # iterate over images
        for idx, path in enumerate(self.images):
            images = ImageReader(path).read_2d_images()
            for filename, image in images.items():
                # measure compression time while compressing
                t0 = time()
                self.compression_method(image)
                t1 = time()
                compression_time = t1 - t0

                # save compressed file
                compressed_path = f"{self.out_path}/{idx}/{filename}-compressed.dcm"
                image.save_as(compressed_path)

                # Medical Image Quality Assessment (MIQA)
                miqa = Miqa(image, dcmread(compressed_path))
                ssim, fsim = miqa.get_miqa_metrics()

                # Compression Assessment
                cr = Path(f"{path}/{filename}").stat().st_size / Path(
                    compressed_path).stat().st_size

                # add to lists
                t_list.append(compression_time)
                cr_list.append(cr)
                ssim_list.append(ssim)
                fsim_list.append(fsim)

        return {"t": t_list, "cr": cr_list, "ssim": ssim_list,
                "fsim": fsim_list}

    @staticmethod
    def final_sample_assessment(metrics):
        """method for FSA

        Args:
            metrics (dict): Dictionary with all measured metrics as keys and the list as value.

        Returns:
            pd.DataFrame: DataFrame with all descriptive statistics for all metrics
        """
        metric_df = pd.DataFrame.from_dict(metrics)
        return metric_df.describe()
