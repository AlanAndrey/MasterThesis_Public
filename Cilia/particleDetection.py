import numpy as np
import pandas as pd
import types
import scipy.signal as signal

"""
Features functions used for particle detection:
- rescale_array
- locate_particles
"""


def rescale_array(array: np.ndarray, new_min: float, new_max: float, dtype: np.dtype) -> np.ndarray:
    """
    Rescale an array to a new range.

    Parameters
    ----------
    array : np.ndarray --The array to rescale.
    new_min : float --The new minimum value.
    new_max : float --The new maximum value.
    dtype : np.dtype --The datatype of the new array.

    Returns
    -------
    np.ndarray --The rescaled array.
    """
    assert new_min < new_max, \
        "New minimum must be smaller than new maximum."
    assert array.min() < array.max(), \
        "Array must have a range greater than zero."

    norm = (array - array.min()) / (array.max() - array.min())
    new = norm * (new_max - new_min) + new_min

    return new.astype(dtype)


def locate_particles(image: np.ndarray, threshold: float, kernel_size: int, particle_size:int, kernel=None, dark_particles = False) -> np.ndarray:
    """
    Locate particles in an image using cross corelation with a kernel.

    Parameters
    ----------
    image : np.ndarray --The image to search for particles. Gets rescaled to cosist of uint8 integers in the range of 0 to 255.
    threshold : float --The threshold to determine particles.
    kernel_size : int --The width and height of the kernel in pixels, must be even number.
    particle_size : int --The size of the particles in pixels.
    kernel : np.ndarray --The kernel to use for cross correlation.
    dark_particles : bool --Whether the particles are dark on a bright background

    Returns
    -------
    np.ndarray --The result of the cross correlation.
    """
    # ToDo: process resulting array from crosscorrelation to get the particle positions

    assert image.ndim == 2, \
        "Image must have two dimensions."
    assert type(kernel) == types.NoneType or kernel.ndim == 2, \
        "Kernel must have two dimensions or be None."
    assert kernel_size % 2 == 0, \
        "Kernel size must be an even number."
    assert kernel_size > particle_size, \
        "Kernel size must be larger than particle size."
    assert particle_size > 0, \
        "Particle size must be greater than zero."

    KS = kernel_size

    # ensure, that kernel and image both have datatype uint8, may be unnecessary
    if kernel is not None:
        kernel = rescale_array(kernel, 0, 255, np.uint8)
    image = rescale_array(image, 0, 255, np.uint8)

    # flip image incase of dark particles
    if dark_particles:
        image = -image + 255


    # create the kernel if not provided based on the gaussian distribution
    if kernel is None:
        kernel = np.zeros((KS, KS))
        for i in range(KS):
            for j in range(KS):
                kernel[i,j] = np.exp(-((i - KS//2)**2 + (j - KS//2)**2) / (2 * (particle_size/2)**2))

    # subtract mean from image and kernel
    image = image - np.mean(image)
    kernel = kernel - np.mean(kernel)

    result = signal.correlate(image, kernel, mode='valid')

    # apply threshold, could be replaced with e.G. %amount of brightest pixels
    # in order to get a more robust result in case of a single bright maxima
    result = np.where(result > threshold*result.max(), result, 0)
    result = rescale_array(result, -1, 1, np.float64)

    return result, image, kernel