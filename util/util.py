from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

from scipy.interpolate import RectBivariateSpline


def imresize(image, size, interp=None, mode=None):
    """
    Resizes a digital image using bivariate spline approximation.
    """
    shape = image.shape
    m, n = shape[0], shape[1]

    if len(shape) == 3 and shape[-1] == 3 or shape[-1] == 4:
        channels = [imresize(image[:, :, i], size) for i in range(shape[-1])]
        m_, n_ = channels[0].shape
        resized = np.empty((m_, n_, shape[-1]))
        for i in range(shape[-1]):
            resized[:, :, i] = channels[i]
        return resized

    if isinstance(size, float) or isinstance(size, int):
        X = np.linspace(0, m - 1, int(size * m))
        Y = np.linspace(0, n - 1, int(size * n))

    elif hasattr(size, "__iter__"):
        if len(size) <= 3:
            X = np.linspace(0, m - 1, size[0])
            Y = np.linspace(0, n - 1, size[1])
        else:
            raise Exception("Size not specified correctly")

    interp = RectBivariateSpline(np.arange(m), np.arange(n), image)
    resized = interp(X, Y)

    return resized
# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
