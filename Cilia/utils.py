import os
import tifftools
from tqdm import tqdm
from PIL import Image
from typing import Callable
from trackpy.motion import msd
from pandas import concat as pandas_concat
"""
Utilities centered around tiff files and the trackpy library
"""
def detect_skipped_frames(dir: str, min_skip: int, key_func: Callable[[str], str]):
    """
    Detects dropped frames in a directory of .tiff/.tif files

    Parameters
    ----------
    dir: str
        Directory to scan for missing frames
    min_skip:
        minimal number of dropped frames to trigger an alert
    key_func: Callable[[str], str]
        function to generate a key from the filename, usually a lambda function
    """
    # get number of frames
    files = [f for f in os.listdir(dir) if f.endswith('.tiff') or f.endswith('.tif')]
    files.sort(key=key_func)
    frame_nr = [key_func(file) for file in files]
    delta = [frame_nr[i+1] - frame_nr[i] for i in range(len(frame_nr)-1)]
    for i in range(len(delta)):
        if delta[i] > min_skip:
            print(f'Warning: {delta[i]} frames missing between {files[i]} and {files[i+1]}\n')



def get_sorted_tiff_files(dir: str, key_func: Callable[[str], str] = lambda file: int(file.split('-')[-1].split('.')[0])):
    """
    Return a sorted list of all the .tiff files in adirectory based on the key function

    Parameters
    ----------
    dir: str
        Directory of files
    key_func: Callable[[str], str]
        function to generate a key from the filename, usually a lambda function
    """
    # get a list of all tif and tiff files in the directory
    tif_files = [f for f in os.listdir(dir) if f.endswith('.tiff') or f.endswith('.tif')]
    tif_files.sort(key=key_func)
    # add the full path to the file
    tif_files = [os.path.join(dir, file) for file in tif_files]
    return tif_files

def merge_tiff_files(dir: str, out:str, key_func=lambda file: int(file.split('-')[-1].split('.')[0])):
    """
    Merge all tif files in a directory into a single tif file using tifftools
    with a tqdm progressbar for long files

    Parameters
    ----------
    dir: str
        Directory containing the tif files
    out: str
        Output file including directory
    batch_size: int
        Number of images to process simultaneously
    """

    tiff_files = get_sorted_tiff_files(dir, key_func = key_func)
    # check if image is grayscale:
    images = []
    for file in tqdm(tiff_files, desc="Loading and converting images"):
        img = Image.open(file)
        if img.mode != 'L':  # Check if the image is not already grayscale
            img = img.convert('L')  # Convert to grayscale
        images.append(img)

    # Ensure all images have the same shape
    assert all(image.size == images[0].size for image in images), "All images must have the same shape"

    # Save the first image and append the rest
    images[0].save(out, save_all=True, append_images=images[1:])

def emsd(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=None):
    """Compute the ensemble mean squared displacements of many particles.

    HARD COPY FROM trackpy.motion.emsd with added std!
    see http://soft-matter.github.io/trackpy/v0.6.4/generated/trackpy.motion.emsd.html#trackpy.motion.emsd
    for the original function

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple particles, including
        columns particle, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : Set to True to include <x>, <y>, <x^2>, <y^2>. Returns
        only <r^2> by default.

    Returns
    -------
    Series[msd, index=t] or, if detail=True,
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    ids = []
    msds = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('particle'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, True, pos_columns))
        ids.append(pid)
    msds = pandas_concat(msds, keys=ids, names=['particle', 'frame'])
    results = msds.mul(msds['N'], axis=0).groupby(level=1).mean()  # weighted average
    results = results.div(msds['N'].groupby(level=1).mean(), axis=0)  # weights normalized
    # Above, lagt is lumped in with the rest for simplicity and speed.
    # Here, rebuild it from the frame index.
    if not detail:
        return results.set_index('lagt')['msd']
    # correctly compute the effective number of independent measurements
    results['N'] = msds['N'].sum()

    # add standart deviation
    msds = msds.reset_index().drop(columns=['<x>', '<y>', '<x^2>', '<y^2>'])
    std = msds.groupby('frame')['msd'].std()
    results['std'] = std
    return results