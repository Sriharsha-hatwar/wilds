# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:21:40 2021

@author: JeanMichelAmath
"""

"""Base augmentations operators."""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
# IMAGE_SIZE = 96 # for camelyon17
IMAGE_SIZE = 256 # for iwildcam


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


## Kornia augmentations
from kornia.augmentation import RandomMotionBlur, RandomChannelShuffle, RandomGrayscale, RandomInvert, RandomGaussianBlur
from kornia.filters import BoxBlur, Laplacian, Canny, Sobel
import skimage as sk
from skimage.filters import gaussian
import torch 

def motion_blur(image, severity=2):
  c = [(10, 3), (15, 5), (15, 7), (15, 11), (20, 15)][severity-1]  
  aug = RandomMotionBlur(kernel_size=c[1], direction=c[0], angle=45.)
  if len(image.shape) <=3:
    image = image.unsqueeze(0)
  return aug(image).squeeze(0)

def channel_shuffle(image, severity):
  aug = RandomChannelShuffle()
  if len(image.shape) <=3:
      image = image.unsqueeze(0)
  return aug(image).squeeze(0)

def grey_scale(image, severity):
  aug = RandomGrayscale(p=1.0)
  if len(image.shape) <=3:
    image = image.unsqueeze(0)
  return aug(image).squeeze(0)

def invert(image, severity):
  aug = RandomInvert(p=1.)
  if len(image.shape) <=3:
    image = image.unsqueeze(0)
  return aug(image).squeeze(0)

def laplacian(image, severity=3):
  c = [3, 7, 11, 15, 19][severity-1]
  aug = Laplacian(kernel_size=c)
  if len(image.shape) <=3:
    image = image.unsqueeze(0)
  return aug(image).squeeze(0)

def canny(image, severity):
  aug = Canny()
  if len(image.shape) <=3:
    image = image.unsqueeze(0)
  return aug(image)[0].squeeze(0).repeat(3,1,1)

def sobel(image, severity):
  aug = Sobel()
  if len(image.shape) <=3:
    image = image.unsqueeze(0)
  return aug(image).squeeze(0)

def box_blur(image, severity=1):
  c = [(3,3), (4,4), (5,5), (6,6), (7,7)][severity-1]
  aug = BoxBlur(kernel_size=c)
  if len(image.shape) <=3:
    image = image.unsqueeze(0)
  return aug(image).squeeze(0)  

def gaussian_blur(image, severity=2):  
  c=[(.4,.4), (1.2,1.2), (2.,2.), (2.8,2.8), (3.6,3.6)][severity-1]
  aug = RandomGaussianBlur(kernel_size=(3,3), sigma=c)
  if len(image.shape) <=3:
    image = image.unsqueeze(0)
  return aug(image).squeeze(0)

def shot_noise(x, severity=5):
  # This would't work on every dataset as the normalization constant is the same on every dataset
  # as a result some examples might have negative values, which is incompatible with poisson noise
  # as lambda should be >= 0
  c = [60, 25, 12, 5, 3][severity - 1]
  x = np.array(x) #/ 255.
  x = np.clip(np.random.poisson(x * c) / float(c), 0, 1) #* 255
  return torch.tensor(x, dtype=torch.float32) 

def impulse_noise(x, severity=4):
  c = [.03, .06, .09, 0.17, 0.27][severity - 1]  
  x = np.array(x)
  x = sk.util.random_noise(x, mode='s&p', amount=c)
  x = np.clip(x, 0, 1) 
  return torch.tensor(x, dtype=torch.float32) 

def spatter(x, severity=4):
  c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
        (0.65, 0.3, 3, 0.68, 0.6, 0),
        (0.65, 0.3, 2, 0.68, 0.5, 0),
        (0.65, 0.3, 1, 0.65, 1.5, 1),
        (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]

  x = np.array(x, dtype=np.float32) #/ 255.

  liquid_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])

  liquid_layer = gaussian(liquid_layer, sigma=c[2])
  liquid_layer[liquid_layer < c[3]] = 0
  
  m = np.where(liquid_layer > c[3], 1, 0)
  m = gaussian(m.astype(np.float32), sigma=c[4])
  m[m < 0.8] = 0

  # mud spatter
  color = 63 / 255. * np.ones_like(x) * m
  x *= (1 - m)
  x = np.clip(x + color, 0, 1) #* 255  
  return torch.tensor(x, dtype=torch.float32) 

def gaussian_noise(x, severity=5):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) #/ 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) #* 255
    return torch.tensor(x, dtype=torch.float32) 

def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def fog(x, severity=5):
  c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

  x = np.array(x) #/ 255.
  max_val = x.max()
  x = x + c[0] * plasma_fractal(wibbledecay=c[1])[:IMAGE_SIZE, :IMAGE_SIZE]
  x = np.clip(x * max_val / (max_val + c[0]), 0, 1) #* 255
  return torch.tensor(x, dtype=torch.float32) 

augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_pil = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, contrast, brightness, sharpness, color
]

augmentations_kornia = [
    gaussian_noise, spatter, canny, gaussian_blur,
    motion_blur, box_blur, laplacian, channel_shuffle, invert,
    grey_scale, sobel, impulse_noise
]

augmentations_all = augmentations_pil + augmentations_kornia


