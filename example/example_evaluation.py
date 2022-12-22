from pydicom.uid import RLELossless

from src.medical_image_compression.evaluation import Evaluator


def compress(image):
    image.compress(RLELossless)


method = compress

evaluator = Evaluator(["/home/hendrik/Dokumente/images/CT-1", "/home/hendrik/Dokumente/images/CT-2",
                       "/home/hendrik/Dokumente/images/CT-3", "/home/hendrik/Dokumente/images/CT-4",
                       "/home/hendrik/Dokumente/images/CT-5", "/home/hendrik/Dokumente/images/CT-6"],
                      compression_method=method, out_path="/home/hendrik/Dokumente/images_compressed")

t = evaluator.evaluate()
print(t)
