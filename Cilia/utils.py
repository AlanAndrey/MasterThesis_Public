import os
import tifftools
from PIL import Image
from typing import Callable
"""
Utilities centered around tiff files
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
    images = [Image.open(file) for file in tiff_files]
    images[0].save(out, save_all=True, append_images=images[1:])
