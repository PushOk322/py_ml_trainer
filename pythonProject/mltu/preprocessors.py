import os
import typing
import numpy as np
import logging

from . import Image

""" Implemented Preprocessors:
- ImageReader - Read image from path and return image and label
- ImageCropper - Crop image to (width, height)
"""

class ImageReader:
    """Read image from path and return image and label"""
    def __init__(self, image_class: Image, log_level: int = logging.INFO, ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self._image_class = image_class

    def __call__(self, image_path: typing.Union[str, np.ndarray], label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Read image from path and return image and label
        
        Args:
            image_path (typing.Union[str, np.ndarray]): Path to image or numpy array
            label (Any): Label of image

        Returns:
            Image: Image object
            Any: Label of image
        """
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
        elif isinstance(image_path, np.ndarray):
            pass
        else:
            raise TypeError(f"Image {image_path} is not a string or numpy array.")

        image = self._image_class(image=image_path)

        if not image.init_successful:
            image = None
            self.logger.warning(f"Image {image_path} could not be read, returning None.")

        return image, label

