import tensorflow as tf
import tensorflow.keras as kr
import cv2
from os import listdir
from os.path import isfile, join
import os
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob 
import skimage.draw
import xmltodict
import json
from skimage import io
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
def load_image():
  image = io.imread(path)
  print(image.dtype)
  print(type(image))
  print(image.ndim)
  # If grayscale. Convert to RGB for consistency.
  if image.ndim != 3:
      image = skimage.color.gray2rgb(image)
  if image.shape[-1] == 4:
      image = image[..., :3]
  original_shape = image.shape
  image, window, scale, padding, crop = utils.resize_image(
      image,
      min_dim=config.IMAGE_MIN_DIM,
      min_scale=config.IMAGE_MIN_SCALE,
      max_dim=config.IMAGE_MAX_DIM,
      mode=config.IMAGE_RESIZE_MODE)
  return image

x = load_image('./ezgif.com-gif-maker.jpg')

