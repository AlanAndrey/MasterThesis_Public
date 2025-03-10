import pims
import math
import numpy as np
import pandas as pd
import trackpy as tp
import trackpy.diag as tpdiag
import scipy.constants as sc
from scipy.special import gamma
import re

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .utils import detect_skipped_frames, get_sorted_tiff_files, emsd
from .nd2_reader import ND2Reader, ND2Sequence

def biggest_factors(x:int)->tuple:
    """
    Calculate the biggest two factors of a number

    Parameters
    ----------
    x : int
        The number to calculate the factors of

    Returns
    -------
    tuple
        The two biggest factors of the number
    """
    a = math.isqrt(x)
    while x % a != 0:
        a -= 1
    return (a, x // a)

"""
Class for tracking particles in cilia videos
"""

class ParticleTracker:
    """
    Class for tracking particles based on a series of monochromatic images
    """
    def __init__(self):
        self.frames = None # the frames to analyze
        self.file_dir= None # the directory of the frames
        self.last_located = None # the DataFrame of the last located particles
        self.tracks = None # the DataFrame of the particles identified
        self.trajectory = None # the DataFrame of the linked particles
        self.length = None # the length of the video
        self.emsd = None # the DataFrame of the mean squared displacement
        self.results = None # table of all frequency dependent results
        self.FPS = None # the frames per second of the video
        self.mpp = None
        tp.quiet() # suppress trackpy warnings

    def enable_tp_warnings(self)->'ParticleTracker':
        """
        Enable trackpy warnings
        """
        tp.quiet(False)
        return self

    def performance_report(self)->'ParticleTracker':
        """
        Print the performance report of trackpy
        """
        print("Trackpy Performance Report:")
        tpdiag.performance_report()
        return self

    def load(self, file_dir:str, format:str, fps:int=None)->'ParticleTracker':
        """
        Load a series of grayscale images from a directory

        Parameters
        ----------
        file_dir : str
            The directory containing the images or image file
        format : str
            The file format of the images using Wildcards, e.g. *.tiff
            see

            https://support.microsoft.com/en-us/office/examples-of-wildcard-characters-939e153f-bd30-47e4-a763-61897c87b3f4
        fps : int
            The frames per second of the video

        Returns
        -------
        True if loaded sucessfully, otherwise the exception is raised
        """

        assert 'nd2' in format or isinstance(fps, (int, float)), 'Invalid fps type, must be integer or float'
        self.FPS = fps
        # check if a single file is loaded or many
        if '*' in format:
            # detect skipped frames
            def get_nr(file):
                # return last number in file name
                return int(re.findall(r'\d+', file)[-1])
            detect_skipped_frames(file_dir, 1, get_nr)
            images = get_sorted_tiff_files(file_dir, get_nr)

            self.frames = pims.ImageSequence(images)
        # custom loading for nd2 files, still in pims format but whole file is loaded
        elif 'nd2' in format:
            reader = ND2Reader(file_dir+'/'+format)
            self.FPS = reader.get_fps()
            self.mpp = reader.get_mpp()
            self.frames = ND2Sequence(file_dir+'/'+format)
        else:
            self.frames = pims.TiffStack(file_dir+'/'+format)
        self.file_dir = file_dir
        self.shape = self.frames[0].shape

        return self

    def __len__(self):
        return self.frames.__len__()

    # enable fram grab via array index
    def __getitem__(self, key):
        return self.frames[key]

    def crop(self, x:tuple, y:tuple)->'ParticleTracker':
        """
        Crop the video to a specific region

        Parameters
        ----------
        x : tuple
            The vertical coordinates of the region to crop
        y : tuple
            The horizontal coordinates of the region to crop
        """
        assert self.frames is not None, 'No frames loaded'
        assert x[0] < x[1], 'Invalid x coordinates'
        assert y[0] < y[1], 'Invalid y coordinates'
        assert x[1] <= self.shape[0], 'x coordinates out of range'
        assert y[1] <= self.shape[1], 'y coordinates out of range'

        x_new = self.shape[0]-x[1]
        y_new = self.shape[1]-y[1]

        self.frames = pims.process.crop(self.frames, ((x[0],x_new), (y[0],y_new)))
        self.shape = self.frames[0].shape
        return self

    def stride(self, stride:int)->'ParticleTracker':
        """
        Reduce the number of frames by a specific factor

        Parameters
        ----------
        stride : int
            The factor to reduce the number of frames by
        """
        assert self.frames is not None, 'No frames loaded'
        assert stride > 0, 'Invalid stride'

        self.frames = self.frames[::stride]
        self.FPS = self.FPS/stride
        return self

    def cut(self, start:int, end:int)->'ParticleTracker':
        """
        Cut the video to a specific range of frames

        Parameters
        ----------
        start : int
            The first frame to include
        end : int
            The last frame to include
        """
        assert self.frames is not None, 'No frames loaded'
        assert start < end, 'Invalid frame range'
        assert end < len(self), 'End frame out of range'

        self.frames = self.frames[start:end]
        return self

    def invert(self):
        """
        Invert the grayscale values of the frames
        """
        assert self.frames is not None, 'No frames loaded'

        @pims.pipeline
        def invert(image):
            return 255 - image

        self.frames = invert(self.frames)
        return self

    def transpose(self):
        """
        Transpose the frames
        """
        assert self.frames is not None, 'No frames loaded'

        @pims.pipeline
        def transpose(image):
            return image.T

        self.frames = transpose(self.frames)
        return self

    def scale_full(self):
        """
        Scale the frames to the full range of the datatype
        """
        assert self.frames is not None, 'No frames loaded'

        _min, _max = np.inf, -np.inf
        for frame in self.frames:
            _min = np.minimum(_min, frame.min())
            _max = np.maximum(_max, frame.max())

        dtype = self.frames[0].dtype
        _range = np.iinfo(dtype).max

        @pims.pipeline
        def scale(image):
            return ((image - _min) / (_max - _min) * _range).astype(dtype)

        self.frames = scale(self.frames)
        return self

    def subtract_mean(self):
        """
        Subtract the mean image from all images. This is implemented as a additional
        step in the PIMS pipeline.
        """

        assert self.frames is not None, 'No frames loaded'

        #get the mean image
        mean_image = np.zeros(self.frames[0].shape)
        min_value = np.inf
        max_value = -np.inf
        for frame in self.frames:
            mean_image += frame
            min_value = min(min_value, frame.min())
            max_value = max(max_value, frame.max())

        mean_image /= len(self)

        # dtype of image to rescale into
        dtype = self.frames[0].dtype
        _range = np.iinfo(dtype).max

        #create the pipeline function to subtract the mean and scale the values to 0 up to dtype max
        @pims.pipeline
        def subtract_mean_pil(image):
            return ((image - mean_image + min_value) / (max_value - min_value)* _range).astype(dtype)

        self.frames = subtract_mean_pil(self.frames)
        return self

    def show_frame(self, frames_list, size=(10,10), area=None)->'ParticleTracker':
        """
        Display frames from a list in a grid layout.

        Parameters
        ----------
        frames_list :list
            A list of frames to be displayed.
        size :tuple, optional
            The size of the figure. Defaults to (10, 10).
        area :tuple of tuples, optional
            The area to crop the frames to. Defaults to None.
        """
        assert self.frames is not None, 'No frames loaded'
        assert max(frames_list) < len(self.frames), 'Frame number out of range'
        assert area is None or isinstance(area, tuple), 'Invalid area'

        # calculate the biggest two factors of the number of frames
        (a, b) = biggest_factors(len(frames_list))

        # plot the frames
        fig, axs = plt.subplots(a, b, figsize=size)
        if isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for i, ax in enumerate(axs.flat):
            if area is not None:
                im = ax.imshow(self.frames[frames_list[i]][area[0][0]:area[0][1], area[1][0]:area[1][1]], cmap='gray', norm=Normalize(vmin=0, vmax=255))
            else:
                im = ax.imshow(self.frames[i], cmap='gray')
            ax.set(title=f'Frame {frames_list[i]%len(self)}')
            _ = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.show()
        return self

    def locate_in_frame(self, frame:int, tp_locate_kwargs:dict, show=False)->'ParticleTracker':
        """
        Locate particles in a specific frame using trackpy

        Parameters
        ----------
        frame : int
            The frame number to locate particles in
        tp_locate_kwargs : dict
            Dict of all kwargs supported by trackpy.locate

            'diameter' is required!, the diameter of the particles in pixels, see

            https://soft-matter.github.io/trackpy/dev/generated/trackpy.locate.html?highlight=locate#trackpy.locate
        show : bool, optional
            Whether to display the located particles. Defaults to False.
        """
        assert self.frames is not None, 'No frames loaded'
        assert frame < len(self.frames), 'Frame number out of range'

        # locate particles using trackpy
        particles = tp.locate(self.frames[frame], **tp_locate_kwargs)
        self.last_located = (frame, particles)

        # show the located particles
        if show:
            self.show_located()
        return self

    def show_located(self, size=(20,10))->'ParticleTracker':
        """
        Display the last located particles in the frame used to locate them

        Parameters
        ----------
        size : tuple, optional
            The size of the figure. Defaults to (10, 10).
        """
        assert self.last_located is not None, 'No particles located'

        frame = self.last_located[0]
        particles = self.last_located[1]
        _, axes = plt.subplot_mosaic([['a', 'b'], ['a', 'c'], ['a', 'd']], figsize=size)
        # imshow
        axes['a'].imshow(self.frames[frame], cmap='gray')
        axes['a'].scatter(particles['x'], particles['y'], s=100*particles['size'],
                   c='none', edgecolors='r')
        axes['a'].set(title=f'Frame {frame%len(self)}')
        # histogramm of the particle masses
        axes['b'].hist(particles['raw_mass'], bins=50)
        axes['b'].set(xlabel='mass', ylabel='count')
        # histogram of the particle sizes
        axes['c'].hist(particles['size'], bins=50)
        axes['c'].set(xlabel='size', ylabel='count')
        # histogram of the particle eccentricities
        axes['d'].hist(particles['ecc'], bins=50)
        axes['d'].set(xlabel='eccentricity', ylabel='count')
        plt.show()
        return self

    def __reset_index__(self, df):
        """
        Sanitized the DataFrame which is returned by various trackpy functions
        """
        if isinstance(df.index, pd.RangeIndex):
            df = df.copy()
            df = df.drop(columns=[i for i in df.index.names if i in df.columns])
            df = df.reset_index()
        return df

    def __update_weights__(self, df):
        """
        Update the weights of the trajectorys based on the number of frames a particle is tracked
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame of the trajectorys
        """
        assert df is not None, 'No trajectorys assembled'

        # drop all columns exc ept particle and frame
        df = df.drop(columns=[col for col in df.columns if col not in ['particle']])

        # need to sanitize input since trackpy creates multilevel indexing sometimes
        frame_particle_df = self.__reset_index__(df)

        # get the duration of each trajectory
        grouped = frame_particle_df.groupby('particle')
        durations = grouped['frame'].max() - grouped['frame'].min()

        # get weights for the trajectorys timesteps
        weights = np.zeros(durations.max() + 1)
        for duration in durations:
            weights[:duration + 1] += 1
        self.weights = weights

    def assemble_tracks(self, tp_locate_kwargs:dict, tp_link_kwargs:dict,
                stub_treshold:int, show=False)->'ParticleTracker':
        """
        Tracks particles throughout all frames and links them using trackpy

        Parameters
        ----------
        tp_locate_kwargs : dict
            Dict of all kwargs supported by trackpy.locate, see

            https://soft-matter.github.io/trackpy/dev/generated/trackpy.batch.html
        tp_link_kwargs : dict
            Dict of all kwargs supported by trackpy.link_df, see

            http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.link_df.html
        stub_treshold : int
            The minimal number of frames a particle must be tracked to be
            included in the trajectory
        show : bool, optional
            Whether to display the linked trajectorys. Defaults to False.
        """
        assert self.frames is not None, 'No frames loaded'
        assert 'diameter' in tp_locate_kwargs, 'diameter is required'

        # track particles using trackpy
        self.tracks = tp.batch(self.frames, **tp_locate_kwargs)
        self.trajectory = tp.link_df(self.tracks, **tp_link_kwargs)
        self.trajectory = tp.filter_stubs(self.trajectory, stub_treshold)

        if show:
            self.show_trajectorys()
        return self

    def show_trajectorys(self, size=(5,5))->'ParticleTracker':
        """
        Display the linked trajectorys

        Parameters
        ----------
        size : tuple, optional
            The size of the figure. Defaults to (10, 10).
        """
        assert self.trajectory is not None, 'No trajectorys assembled'

        _, ax = plt.subplots(1, 1, figsize=size)
        for _, group in self.trajectory.groupby('particle'):
            ax.plot(group['x'], group['y'], '-', linewidth=0.5)
        ax.set(title='Trajectories', xlabel='x [px]', ylabel='y [px]')
        plt.show()
        return self

    def subtract_drift(self, show=False)->'ParticleTracker':
        """
        Subtract the drift from the trajectorys using trackpys compute and subtract drift function
        compute: http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.compute_drift.html
        subtract: http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.subtract_drift.html
        """
        assert self.trajectory is not None, 'No trajectorys assembled'

        # compute the drift
        drift = tp.compute_drift(self.trajectory)
        # subtract the drift
        self.trajectory = tp.subtract_drift(self.trajectory, drift)

        if show:
            self.show_drift(drift)
        return self

    def show_drift(self, drift, size=(5,5))->'ParticleTracker':
        """
        Display the drift of the trajectorys

        Parameters
        ----------
        size : tuple, optional
            The size of the figure. Defaults to (10, 10).
        """
        assert self.trajectory is not None, 'No trajectorys assembled'

        _, ax = plt.subplots(1, 1, figsize=size)
        ax.plot(drift['x'], 'k', label='x')
        ax.plot(drift['y'], 'r', label='y')
        ax.set(title='Drift', xlabel='frame', ylabel='drift [px]')
        ax.legend()
        plt.show()

    def calculate_msd(self, tp_emsd_kwargs:dict, percentile=50, show=False)->'ParticleTracker':
        """
        Calculate the mean squared displacement of the linked trajectorys using
        trackpys emsd function

        Parameters
        ----------
        tp_emsd_kwargs : dict
            Dict of all kwargs supported by trackpy.emsd, see

            http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.emsd.html
        percentile: float, optional
            The length of the msd kept for further processing in percent. Defaults to 100.
            This length is determined by the number of particles tracked for each timestep
        show : bool, optional
            Whether to display the MSD plot. Defaults to False.
        """
        assert self.trajectory is not None, 'No trajectorys assembled'
        assert 'max_lagtime' in tp_emsd_kwargs, 'max_lagtime is set to 100 per default'
        # set the frames per second if not set by the user
        if 'fps' not in tp_emsd_kwargs:
            tp_emsd_kwargs['fps'] = self.FPS
            #overwrite the microns per pixel if present from nd2 file
        if self.mpp is not None:
            tp_emsd_kwargs['mpp'] = self.mpp

        # units of result are in seconds and microns
        self.trajectory = self.__reset_index__(self.trajectory)

        _emsd = emsd(self.trajectory, **tp_emsd_kwargs, detail=True)

        # update the weights of the trajectorys
        self.__update_weights__(self.trajectory.set_index('frame'))

        # cut the percentile of the msd
        _max = self.weights[0]
        _min = self.weights[-1]
        cut = _max*(100-percentile)/100
        if cut < _min:
            cut = _min
        idx = np.nonzero(self.weights >= cut)[0][-1]

        # cut emsd and weights
        self.emsd = _emsd.iloc[:idx]
        self.weights = self.weights[:idx]


        if show:
            self.show_msd()
        return self

    def show_msd(self, size=(5,5))->'ParticleTracker':
        """
        Display the mean squared displacement

        Parameters
        ----------
        size : tuple, optional
            The size of the figure. Defaults to (10, 10).
        """
        assert self.emsd is not None, 'No MSD calculated'

        _, ax = plt.subplots(1, 1, figsize=size)
        ax.scatter(self.emsd['lagt'], self.emsd['msd'], s=2, c='k')
        ax.set(xlabel='lag time [s]', ylabel=r'$\langle \Delta r^2 \rangle [\mu m^2]$',
               title='Mean Squared Displacement')
        ax.fill_between(self.emsd['lagt'], self.emsd['msd'] - 0.1*self.emsd['std'],
                        self.emsd['msd'] + 0.1*self.emsd['std'], alpha=0.5, color='r', label=r'$0.1\sigma$')
        ax.legend(loc='upper left')
        plt.show()
        return self

    def __unilateral_fourier__(self, show=False)->'ParticleTracker':
        """
        Calculate the unilateral Fourier transform of the msd

        Parameters
        ----------
        show : bool, optional
            Whether to display the Fourier transform. Defaults to False.
        """
        assert self.emsd is not None, 'No msd calculated jet'

        # calculate the Fourier transform, note conversion between μm^2 and m^2
        fourier = np.fft.fft(self.emsd['msd']*1e-12)
        freq = np.fft.fftfreq(len(self.emsd['msd']), d=self.emsd['lagt'][2]-self.emsd['lagt'][1])
        # cut negative frequencies and zero frequency
        idx = len(fourier)//2
        fourier = fourier[1:idx]
        freq = freq[1:idx]

        # assemble the results table
        if self.results is None:
            self.results = pd.DataFrame({'freq [Hz]': freq, 'F(msd) [m*s]': fourier})
        else:
            self.results['F(msd) [m*s]'] = fourier

        if show:
            self.show_fourier()

    def show_fourier(self, size=(5,5))->'ParticleTracker':
        """
        Display the unilateral Fourier transform

        Parameters
        ----------
        freq : np.array
            The frequencies of the Fourier transform
        fourier : np.array
            The Fourier transform
        size : tuple, optional
            The size of the figure. Defaults to (10, 10).
        """
        # calculate the power spectrum
        cut = 1
        pwrspec = np.abs(self.results['F(msd) [m*s]'])**2
        _, ax = plt.subplots(1, 1, figsize=size)
        ax.plot(self.results['freq [Hz]'][cut:], pwrspec[cut:], 'k')
        # units of the power spectrum are in m*Hz
        ax.set(xlabel='frequency [Hz]', ylabel='Amplitude', title='Power Spectrum', yscale='log')
        plt.show()
        return self

    def __calculate_laplace__(self):
        """
        Calculate the laplace transform of the msd

        Parameters
        ----------
        s : complex, optional
            Frequency domain parameter s=σ+iω. Defaults to s=0+i.
        show : bool, optional
            Whether to display the laplace transform. Defaults to False.
        """
        assert self.emsd is not None, 'No msd calculated'

        # create frequency domain
        freq = np.fft.fftfreq(len(self.emsd['msd']), d=self.emsd['lagt'][2]-self.emsd['lagt'][1])
        freq = freq[1:len(freq)//2]

        msd = self.emsd['msd'].to_numpy()*1e-12 # convert to m^2
        lagt = self.emsd['lagt'].to_numpy()

        # calculate the laplace transform of the msd
        def single_laplace(s):
            # use the trapezoidal rule to calculate the integral
            # F(s) ≈ Δt/2 * [f(0) + 2 * Σ_{n=1}^{N-1} f(n*Δt) * e^{-s*n*Δt} + f(N*Δt) * e^{-s*N*Δt}]
            res = (lagt[1]-lagt[0])/2 * (msd[0] + 2*np.sum([msd[i] * np.exp(-s*lagt[i]) for i in range(1,len(lagt)-1)]) + msd[-1]*np.exp(-s*lagt[-1]))
            return res

        laplace = np.array([single_laplace(0+1j*s) for s in freq])

        # assemble the results table
        if self.results is None:
            self.results = pd.DataFrame({'freq [Hz]': freq, 'L(msd) [m*s]': laplace})
        else:
            self.results['L(msd) [m*s]'] = laplace

    def __calculate_polyfit_laplace__(self, coeffs):
        """
        Calculate the laplace transform of a polynomial fit of the msd

        Parameters
        ----------
        coeffs : np.array
            The coefficients of the polynomial fit of the msd
        s : complex, optional
            Frequency domain parameter s=σ+iω. Defaults to s=0+i.
        show : bool, optional
            Whether to display the laplace transform. Defaults to False.
        """
        assert self.emsd is not None, 'No msd calculated'

        # create frequency domain
        freq = np.fft.fftfreq(len(self.emsd['msd']), d=self.emsd['lagt'][2]-self.emsd['lagt'][1])
        freq = freq[1:len(freq)//2]

        # calculate the laplace transform of the polynomial fit
        def single_laplace(s):
            return np.sum([coeffs[i] * math.factorial(i) / s**(i+1) for i in range(len(coeffs))])

        laplace = np.array([single_laplace(0+1j*s) for s in freq])
        # assemble the results table
        if self.results is None:
            self.results = pd.DataFrame({'freq [Hz]': freq, 'L(poly(msd)) [m*s]': laplace})
        else:
            self.results['L(poly(msd)) [m*s]'] = laplace


    def calculate_moduli(self, temperature, radius, modus='viscoelastic', show=False)->'ParticleTracker':
        """
        Calculate the complex, viscous and elastic modulus from the fourier
        transformed msd.

        Parameters
        ----------
        temperature : float
            The temperature of the fluid in Kelvin
        radius : float
            The radius of the particles in meters
        modus : str, optional
            Either visocelastic per default o viscous for purely viscous fluids.
        show : bool, optional
            Whether to display the moduli. Defaults to False.
        """
        assert modus in ['viscoelastic', 'viscous', 'laplace', 'powerlaw'] or  'poly_laplace' in modus, 'Invalid modus'


        K = sc.Boltzmann # m^2*kg*s^-2*K^-1
        PI = sc.pi

        # viscoelastic moduli
        if modus == 'viscoelastic':
            self.__unilateral_fourier__()

            # calculate the complex moduli
            # units: Pa or kg*m^-1*s^-2
            g = 2*K*temperature / (3*PI*radius * 1j * self.results['freq [Hz]'] * self.results['F(msd) [m*s]'])
            # calculate the viscous and elastic moduli
            g_el = np.abs(np.real(g)) # Pa
            g_vis = np.abs(np.imag(g)) # Pa

            # append the results to the moduli results table
            self.moduli = pd.DataFrame({'freq [Hz]': self.results['freq [Hz]'],
                                        'G* [Pa]': g,
                                        'G\' [Pa]': g_el,
                                        'G\" [Pa]': g_vis})
            if show:
                self.show_moduli(scale='linear')

        # purely viscous modulus
        elif modus == 'viscous':
            # create linear fit of the msd
            fit, [_, _, _, _] = np.polynomial.polynomial.polyfit(self.emsd['lagt'], self.emsd['msd'], 1, w=self.weights, full=True)
            # fit[0] = offset [dr^2], fit[1] = dr/t [um^2/t]
            # calculate the viscous modulus using the t/dr relation
            n = 2 * K * temperature / (3 * PI * radius) / fit[1] * 1e12 # Pa*s

            print(f'Viscous Modulus: {n} Pa*s')

        elif modus == 'laplace':
            # first create laplace transform of the msd
            self.__calculate_laplace__()

            # calculatet the complex moduli
            g = 2*K*temperature / (3*PI*radius * 1j * self.results['freq [Hz]'] * self.results['L(msd) [m*s]'])
            g_el = np.abs(np.real(g)) # Pa
            g_vis = np.abs(np.imag(g)) # Pa

            self.moduli = pd.DataFrame({'freq [Hz]': self.results['freq [Hz]'],
                                        'G* [Pa]': g,
                                        'G\' [Pa]': g_el,
                                        'G\" [Pa]': g_vis})
            if show:
                self.show_moduli()

        elif 'poly_laplace' in modus:
            # extract degree of polynomial as only number in string
            deg = int(re.findall(r'\d+', modus)[0])
            # create a polynomial fit of the msd
            fit, [_, _, _, _] = np.polynomial.polynomial.polyfit(self.emsd['lagt'], self.emsd['msd']*1e-12, deg, w=self.weights, full=True)

            # next calculate the analytical laplace transform of the polynomial:
            self.__calculate_polyfit_laplace__(fit)
            g = 2*K*temperature / (3*PI*radius * 1j * self.results['freq [Hz]'] * self.results['L(poly(msd)) [m*s]'])
            g_el = np.abs(np.real(g)) # Pa
            g_vis = np.abs(np.imag(g)) # Pa

            self.moduli = pd.DataFrame({'freq [Hz]': self.results['freq [Hz]'],
                                        'G* [Pa]': g,
                                        'G\' [Pa]': g_el,
                                        'G\" [Pa]': g_vis})
            if show:
                self.show_moduli()

        elif modus == 'powerlaw':
            # create a rolling powerlaf fit of the msd MSD = A * (t/t_0)^alpha(t_0)
            # in order to extract alpha we only need to fit the slope of the log log plot

            window_size = 7 # the number of points to consider for each fit on the log log plot, crop at the edges
            log_msd = np.log(self.emsd['msd'])
            log_lagt = np.log(self.emsd['lagt'])
            alpha = np.zeros(len(log_msd))
            for t_0 in range(len(log_msd)):
                idx = t_0 - window_size//2
                if idx < 0:
                    idx = 0
                if idx + window_size > len(log_msd):
                    idx = len(log_msd) - window_size
                fit = np.polyfit(log_lagt[idx:idx+window_size], log_msd[idx:idx+window_size], 1, w=self.weights[idx:idx+window_size])
                alpha[t_0] = fit[0]

            # calculate the complex moduli
            prefactor = 2 * K * temperature / (3 * PI * radius)
            g = prefactor / (self.emsd['msd'].to_numpy()*1e-12 * np.array(gamma(1 + alpha)))
            g_el = g * np.cos(PI/2 * alpha)
            g_vis = g * np.sin(PI/2 * alpha)

            self.moduli = pd.DataFrame({'freq [Hz]': 1/self.emsd['lagt'],
                                        'G* [Pa]': g,
                                        'G\' [Pa]': g_el,
                                        'G\" [Pa]': g_vis})
            if show:
                self.show_moduli()


        return self

    def show_moduli(self, size=(5,5), scale='log')->'ParticleTracker':
        """
        Display the viscous and elastic moduli

        Parameters
        ----------
        size : tuple, optional
            The size of the figure. Defaults to (10, 10).
        """
        assert self.results is not None, 'No moduli calculated'

        _, ax = plt.subplots(1, 1, figsize=size)
        ax.scatter(self.moduli['freq [Hz]'], self.moduli['G\' [Pa]'], s=1, label='G\'')
        ax.scatter(self.moduli['freq [Hz]'], self.moduli['G\" [Pa]'], s=1, label='G\"')
        ax.set(xlabel='frequency [Hz]', ylabel='Modulus [Pa]',
               title='Viscous & Elastic Modulus', yscale='log', xscale=scale)
        ax.legend()
        plt.show()
        return self
