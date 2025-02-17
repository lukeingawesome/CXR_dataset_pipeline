import os
import pathlib
import omegaconf
from threading import Thread
import albumentations as A
from typing import Callable, Iterable, Literal, Mapping, Union
import cv2
import numpy as np
import shapely
from typing import Any, Dict
import logging

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig, ListConfig

# from .randaug_pixel import random_pixel_augment_v2
# from .randaug_spatial import random_resize_crop, random_spatial_augment_v1, random_spatial_augment_v2

# Configure logging
logger = logging.getLogger(__name__)

# Constants
_EPS = 1e-6
_SUPPORTED_CHANNELS = (3, 4)
_SUPPORTED_COLOR_CONVERSIONS = {
    3: cv2.COLOR_BGR2GRAY,
    4: cv2.COLOR_BGRA2GRAY
}

_AVAILABLE_TRANSFORMS = {
    'resize': A.Resize,
    'random_flip': A.HorizontalFlip,
    'to_tensor_v2': ToTensorV2,
}

def build_transforms(transforms: Dict[str, Any]) -> list:
    """Build a list of albumentation transforms from config dictionary.
    
    Args:
        transforms: Dictionary of transform configurations
        
    Returns:
        List of instantiated albumentation transforms
    """
    def _process_transform_params(transform: Dict[str, Any]) -> dict:
        return {
            key: tuple(val) if isinstance(val, ListConfig)
            else _process_transform_params(val) if isinstance(val, DictConfig)
            else val
            for key, val in transform.items()
        }

    return [
        _AVAILABLE_TRANSFORMS[str(key)](**_process_transform_params(dict(val) if val else {}))
        for key, val in transforms.items()
    ]

def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """
    Applies min-max normalization to the image.
    image = (image - image.min()) / (image.max() - image.min())
    Args:
        image (np.ndarray): the image to do normalizaion

    Returns:
        image that was normalized.
    """
    image = image.astype(np.float32)

    min_val, max_val = image.min(), image.max()
    image = (image - min_val) / (max_val - min_val + _EPS)
    return image


class LoadCXRImageError(Exception):
    pass


def windowing(image: np.ndarray, use_median: bool = False,
              width_param: float = 4.0, brightness: float = 0.0) -> np.ndarray:
    """
    Windowing function that clips the values based on the given params.
    Args:
        image (str): the image to do the windowing
        use_median (bool): use median as center if True, mean otherwise
        width_param (float): the width of the value range for windowing.
        brightness (float) : brightness_rate. a value between 0 and 1 and indicates the amount to subtract.

    Returns:
        image that was windowed.
    """
    center = np.median(image) if use_median else image.mean()

    if brightness:
        center *= 1 - brightness

    range_width_half = (image.std() * width_param) / 2.0
    low = center - range_width_half
    high = center + range_width_half
    image = np.clip(image, low, high)
    return image


def get_transform(
    transforms: Union[dict, A.Compose, None], 
    **kwargs
) -> Union[A.Compose, Callable]:
    """Get albumentation transform composition from config or existing transform.
    
    Args:
        transforms: Transform configuration or existing composition
        **kwargs: Additional arguments for A.Compose
        
    Returns:
        Composed transform or identity function if transforms is None
    """
    if transforms is None:
        logger.warning("No transforms provided; using identity function")
        return lambda x: x
    
    if isinstance(transforms, (dict, omegaconf.DictConfig)):
        return A.Compose(transforms=build_transforms(transforms), **kwargs)
    
    if isinstance(transforms, A.Compose):
        return transforms
    
    raise TypeError(f"Unexpected transforms type: {type(transforms)}")

def load_image(
    filename: str, 
    do_windowing: bool = True,
    use_median: bool = True, 
    width_param: float = 4.0
) -> np.ndarray:
    """Load the CXR image and apply windowing & min-max normalization.

    Args:
        filename (str): Name of file to be loaded.
        do_windowing (bool, optional): Whether to apply windowing. Defaults to True.
        use_median (bool, optional): (arg for 'windowing' sub-call) Whether to use_median. Defaults to True.
        width_param (float, optional): (arg for 'windowing' sub-call) Width param for windowing. Defaults to 4.0.

    Raises:
        LoadCXRImageError

    Returns:
        np.ndarray: The image loaded (dtype=np.float32, minmax_normalized: [0., 1.])
    """
    if not pathlib.Path(filename).exists():
        raise LoadCXRImageError(f"Image does not exist: {filename}")

    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise LoadCXRImageError(f"Failed to load image: {filename}")
        
    image = image.astype(np.float32)
    image = image.squeeze()

    if image.ndim >= 3:
        logger.warning(f"Got {image.ndim}-dimensional image, converting to grayscale")
        
        if image.shape[-1] not in _SUPPORTED_CHANNELS:
            raise LoadCXRImageError(
                f"Unsupported number of channels: {image.shape[-1]}, "
                f"supported: {_SUPPORTED_CHANNELS}"
            )
            
        image = cv2.cvtColor(image, _SUPPORTED_COLOR_CONVERSIONS[image.shape[-1]])

    if do_windowing:
        image = windowing(image, use_median, width_param)
    image = min_max_normalize(image)
    
    return np.ascontiguousarray(image)