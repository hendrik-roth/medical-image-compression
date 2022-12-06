import glob

import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread


class ImageReader:
    def __init__(self, path):
        self.path = path

    def read_medical_image(self):
        files = [dcmread(file_name) for file_name in
                 glob.glob(self.path + "/*.dcm", recursive=False)]
        slices = [file for file in files]

        # create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the slices
        for i, s in enumerate(slices):
            img2d = s.pixel_array
            img3d[:, :, i] = img2d
        return slices, img3d
