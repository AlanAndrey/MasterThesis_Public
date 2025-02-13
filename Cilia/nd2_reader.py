import nd2
from pims import FramesSequence, Frame

# custom pims reader for nd2 files, however the whole file has to be loaded into memory
class ND2Sequence(FramesSequence):
    """
    Custom pims reader for a monochromatic nd2 file based on nd2 package.
    see: https://www.talleylambert.com/nd2/ for more information.
    Parameters
    ----------
    path : str
        Path to the nd2 file.
    """
    def __init__ (self, path: str):
        self.file = nd2.ND2File(path)
        assert not self.file.is_rgb, 'RGB files not supported'

    def get_frame(self, i):
        return Frame(self.file.read_frame(i), frame_no=i)

    def __len__(self):
        return self.file.shape[0]

    @property
    def frame_shape(self):
        return self.data.shape[1:]

    @property
    def pixel_type(self):
        return self.file.dtype

    @classmethod
    def class_exts(cls):
        # add nd2 to the list of supported file extensions
        return {'nd2'} | super(ND2Sequence, cls).class_exts()

    def close(self):
        self.file.close()

# Class to read nd2 files and extract useful metadata
class ND2Reader:
    """
    Class to read nd2 files and extract useful metadata.
    """
    def __init__(self, path: str):
        """
        Create a ND2Reader object that holds the following data:
        - shape: shape of the data
        - metadata: metadata of the nd2 file
        - experiment: experiment metadata
        - data: data as numpy array
        - voxel_size: voxel size in microns

        Based on https://www.talleylambert.com/nd2/
        Parameters
        ----------
        path : str
            Path to the nd2 file.
        """
        self.file = nd2.ND2File(path)

        assert not self.file.is_rgb, 'RGB files not supported'

        self.shape = self.file.shape
        self.metadata = self.file.metadata
        self.experiment = self.file.experiment
        self.data = self.file.asarray()
        self.path = self.file.path
        self.voxel_size = self.file.voxel_size(0)
        self.file.close()

    def __len__(self):
        return self.shape[0]

    def get_fps(self):
        assert isinstance(self.experiment[0], nd2.structures.TimeLoop) and len(self.experiment) == 1, 'Data must be single time loop'

        # get avg, min and max time difference between frames in ms
        pd = self.experiment[0].parameters.periodDiff
        avg = pd.avg; min = pd.min; max = pd.max
        assert min >= 0.95 * avg and max <= 1.05 * avg, 'Period difference greater than 5% of average fps'

        return 1 / avg * 1000 #convert to seconds

    def get_mpp(self):
        # get microns per pixel, take x axis as reference since symmetric
        return self.voxel_size[0]
