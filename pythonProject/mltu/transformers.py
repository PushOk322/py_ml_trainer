import cv2
import time
import queue
import typing
import logging
import threading
import importlib
import numpy as np

from . import Image

""" Implemented Transformers:
- ImageResizer - Resize image to (width, height)
- LabelIndexer - Convert label to index by vocab
- LabelPadding - Pad label to max_word_length
"""

class Transformer:
    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
        raise NotImplementedError

class ImageResizer(Transformer):
    """Resize image to (width, height)
    
    Attributes:
        width (int): Width of image
        height (int): Height of image
        keep_aspect_ratio (bool): Whether to keep aspect ratio of image
        padding_color (typing.Tuple[int]): Color to pad image
    """
    def __init__(
        self, 
        width: int, 
        height: int, 
        keep_aspect_ratio: bool=False, 
        padding_color: typing.Tuple[int]=(0, 0, 0)
        ) -> None:
        self._width = width
        self._height = height
        self._keep_aspect_ratio = keep_aspect_ratio
        self._padding_color = padding_color

    @staticmethod
    def unpad_maintaining_aspect_ratio(padded_image: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        height, width = padded_image.shape[:2]
        ratio = min(width / original_width, height / original_height)

        delta_w = width - int(original_width * ratio)
        delta_h = height - int(original_height * ratio)
        left, right = delta_w//2, delta_w-(delta_w//2)
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        unpaded_image = padded_image[top:height-bottom, left:width-right]

        original_image = cv2.resize(unpaded_image, (original_width, original_height))

        return original_image

    @staticmethod
    def resize_maintaining_aspect_ratio(image: np.ndarray, width_target: int, height_target: int, padding_color: typing.Tuple[int]=(0, 0, 0)) -> np.ndarray:
        """ Resize image maintaining aspect ratio and pad with padding_color.

        Args:
            image (np.ndarray): Image to resize
            width_target (int): Target width
            height_target (int): Target height
            padding_color (typing.Tuple[int]): Color to pad image

        Returns:
            np.ndarray: Resized image
        """
        height, width = image.shape[:2]
        ratio = min(width_target / width, height_target / height)
        new_w, new_h = int(width * ratio), int(height * ratio)

        resized_image = cv2.resize(image, (new_w, new_h))
        delta_w = width_target - new_w
        delta_h = height_target - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

        return new_image

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        if not isinstance(image, Image):
            raise TypeError(f"Expected image to be of type Image, got {type(image)}")

        # Maintains aspect ratio and resizes with padding.
        if self._keep_aspect_ratio:
            image_numpy = self.resize_maintaining_aspect_ratio(image.numpy(), self._width, self._height, self._padding_color)
            if isinstance(label, Image):
                label_numpy = self.resize_maintaining_aspect_ratio(label.numpy(), self._width, self._height, self._padding_color)
                label.update(label_numpy)
        else:   
            # Resizes without maintaining aspect ratio.
            image_numpy = cv2.resize(image.numpy(), (self._width, self._height))
            if isinstance(label, Image):
                label_numpy = cv2.resize(label.numpy(), (self._width, self._height))
                label.update(label_numpy)

        image.update(image_numpy)

        return image, label

class LabelIndexer(Transformer):
    """Convert label to index by vocab
    
    Attributes:
        vocab (typing.List[str]): List of characters in vocab
    """
    def __init__(
        self, 
        vocab: typing.List[str]
        ) -> None:
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

class LabelPadding(Transformer):
    """Pad label to max_word_length
    
    Attributes:
        padding_value (int): Value to pad
        max_word_length (int): Maximum length of label
        use_on_batch (bool): Whether to use on batch. Default: False
    """
    def __init__(
        self, 
        padding_value: int,
        max_word_length: int = None, 
        use_on_batch: bool = False
        ) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_word_length is None:
            raise ValueError("max_word_length must be specified if use_on_batch is False")

    def __call__(self, data: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in label])
            padded_labels = []
            for l in label:
                padded_label = np.pad(l, (0, max_len - len(l)), "constant", constant_values=self.padding_value)
                padded_labels.append(padded_label)

            padded_labels = np.array(padded_labels)
            return data, padded_labels

        label = label[:self.max_word_length]
        return data, np.pad(label, (0, self.max_word_length - len(label)), "constant", constant_values=self.padding_value)

