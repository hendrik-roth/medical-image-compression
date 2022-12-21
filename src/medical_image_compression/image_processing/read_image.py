import glob

import numpy as np
from pydicom import dcmread


class ImageReader:
    def __init__(self, path):
        self.path = path

    def read_2d_images(self):
        """read all 2D dicom images inside of a dir

        Returns:
            _type_: _description_
        """
        return {f"{file_name}": dcmread(file_name) for file_name in
                glob.glob(self.path + "/*.dcm", recursive=False)}
