import numpy as np
import matplotlib.pyplot as plt

"""
Features functions and class used for powerspectrum calculation:
- get_powerspectrum
- get_powerspectrum_mean
- Powerspectrum
"""

def subtract_mean(data: np.ndarray)-> np.ndarray:
    """
    Subtract the temporal mean of each pixel

    Parameters
    ----------
    data : np.ndarray --The data array where the last axis is the time axis.
    Returns
    -------
    np.ndarray --The data array with the temporal mean subtracted.
    """
    assert data.ndim >= 1, \
        "Data must have at least one dimension."
    assert data.shape[-1] > 1, \
        "Data must have at least two elements along the time axis."
    return data - np.mean(data, axis=0)


def get_powerspectrum(data:np.ndarray)->np.ndarray:
    """
    Calculates the Powerspectrum of a given data array along the first axis.
    First subtract the temporal mean to remove frequency 0.

    Parameters
    ----------
    data : np.ndarray --The data array where the first axis is the time axis.
    Returns
    -------
    np.ndarray --The powerspectrum for each timeseries.
    """
    assert data.ndim >= 1, \
        "Data must have at least one dimension."
    assert data.shape[-1] > 1, \
        "Data must have at least two elements along the time axis."
    data = subtract_mean(data)
    fft = np.fft.rfft(data, axis=0) # calculate the fft for each timeseries using the rfft
    powerspectrum = np.square(np.absolute(fft)) # calculate the powerspectrum
    return powerspectrum

def get_powerspectrum_mean_density(powerspectrum:np.ndarray, mask:np.ndarray)->np.ndarray:
    """
    Calculates the mean Powerspectrum  density of a given powerspectrum array.

    Parameters
    ----------
    powerspectrum : np.ndarray --The powerspectrum array.
    mask : np.ndarray or None --The mask array, if None, no mask is applied.
    The mask array must have the same shape as the powerspectrum array.
    If values are boolean, the mas is applied by discarding all False values.
    Otherwise, the mask is applied by multiplying the powerspectrum with the mask.

    Returns
    -------
    np.ndarray --The mean powerspectrum density.
    """
    assert powerspectrum.ndim >= 1, \
        "Powerspectrum must have at least one dimension."
    assert powerspectrum.shape[-1] > 1, \
        "Powerspectrum must have at least two elements along the time axis."
    assert mask is None or mask.shape == powerspectrum.shape, \
        "Mask must have the same shape as the powerspectrum."
    if mask is not None:
        if mask.dtype == bool:
            powerspectrum = powerspectrum[mask]
        else:
            powerspectrum *= mask
    pwrspec = np.mean(powerspectrum, axis=(1,2))
    pwrspec /= np.sum(pwrspec)
    return pwrspec

class Powerspectrum:
    def __init__(self, data:np.ndarray):
        """
        Initializes the Powerspectrum object.

        Parameters
        ----------
        data : np.ndarray --The data array where the last axis is the time axis.
        """
        self.data = data

    def show_powerspectrum(self, FPS:float, mask:np.ndarray=None)->None:
        """
        Shows the powerspectrum of the data.

        Parameters
        ----------
        FPS : float --The frames per second of the data.
        mask : np.ndarray or None --The mask array, if None, no mask is applied.
        The mask array must have the same shape as the powerspectrum array.
        If values are boolean, the mas is applied by discarding all False values.
        Otherwise, the mask is applied by multiplying the powerspectrum with the mask.
        """
        powerspectrum = get_powerspectrum(self.data)
        self.mean_powerspectrum = get_powerspectrum_mean_density(powerspectrum, mask)
        # get frequencies for the x-axis, the factor 2 is because of the rfft
        self.freqs = np.fft.rfftfreq(len(self.data), d=1/FPS)
        # avg = np.average(freqs[1:], weights=mean_powerspectrum[1:])
        # std = np.sqrt(np.average((freqs[1:]-avg)**2, weights=mean_powerspectrum[1:]))
        plt.plot(self.freqs[1:], self.mean_powerspectrum[1:])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power density")
        plt.show()
        print('validation10')