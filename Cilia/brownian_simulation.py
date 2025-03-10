import os
import PIL
import numpy as np
from tqdm import tqdm
from datetime import datetime
import scipy.constants as sc

# Readme!
#this code is originally written by Aikaterina Grava and has ben modified to be used in automatic error analysis

class BrownianSimulation:

    def __init__(self, size=(100,100), FPS=30, mpp=100, temp=295,
                 visc=0.001, radius=250e-9, npart=100):
        """
        initialize the simulation object
        Parameters
        ----------
        size: tuple
            size of the simulation box in pixel
        FPS: int
            frames per second
        mpp: int
            pixel size in m/pixel
        temp: float
            temperature of the simulation in K
        visc: float
            viscosity of the medium in Pa*s
        radius: float
            radius of the particles in m
        npart: int
            number of particles to simulate
        """
        self.size = size
        self.FPS = FPS
        self.dt = 1/FPS
        self.mpp = mpp
        self.temp = temp
        self.visc = visc
        self.radius = radius
        self.npart = npart
        self.gen = np.random.default_rng(7)

    def run_simulation(self, output_dir, length, save_pos=False, filename=None):
        """
        Run the simulation and save it to the output directory
        Parameters
        ----------
        output_dir: str
            output directory
        length: int
            number of frames to simulate
        save_pos: bool
            save the position of the particles in the file 'pos_history.npy'
        """
        # calculate the diffusion coefficient
        self.diffusion = sc.k * self.temp / (6 * sc.pi * self.visc * self.radius) # m²/s
        self.diffusion_pixel = self.diffusion / self.mpp**2  # m²/s to pixels²/s

        # define size of particles
        self.part_size = self.radius / self.mpp

        # generate initial positions of particles
        pos_x = self.gen.uniform(0, self.size[0], self.npart)
        pos_y = self.gen.uniform(0, self.size[1], self.npart)
        pos = np.column_stack((pos_x, pos_y))

        # create array to save to if needed
        if save_pos:
            pos_history = np.zeros((length+1, self.npart, 2))
            pos_history[0] = pos

        frames = [self.__draw_frame__(self.size, pos)]

        # run simulation loop
        for i in tqdm(range(length), unit='frames', desc='Running simulation'):
            pos = self.__update_pos__(pos)
            frames.append(self.__draw_frame__(self.size, pos))
            if save_pos:
                pos_history[i+1] = pos

        # create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save the position history
        if save_pos:
            pos_file = filename.split('.')[0] + '_pos_history.npy'
            np.save(os.path.join(output_dir, pos_file), pos_history)

        # save the frames
        curr_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if filename is None:
            filename = f'simulation_{curr_time}.tif'
        images = [PIL.Image.fromarray(frame.T) for frame in frames]
        images[0].save(os.path.join(output_dir, filename), save_all=True, append_images=images[1:])

    def __draw_frame__(self, size, pos):
        """
        Draw a frame of the simulation using the position and size of the particles

        Parameters
        ----------
        size: tuple
            size of the frame
        pos: np.array
            position of the particles
        """

        # create an empty image
        frame = np.zeros(size)

        # draw the particles additively
        for p in pos:
            frame = self.__draw_particle__(frame, p)
        return frame.astype(np.uint8)

    def __draw_particle__(self, frame, pos):
        """
        Draw a particle additively onto a frame using a gaussian intensity distribution
        """
        # check first, if particle out of bounds, then skip frame drawing
        if pos[0] < 0 or pos[0] >= frame.shape[0] or pos[1] < 0 or pos[1] >= frame.shape[1]:
            return frame

        # calculate the intensity for each pixel in the frame via meshgrid
        x, y = np.ogrid[:frame.shape[0], :frame.shape[1]]
        distance_squared_map = (x - pos[0])**2 + (y - pos[1])**2
        intensity = 255*np.exp(-distance_squared_map / (2 * self.part_size**2))

        # add the particle to the frame and return it
        return np.maximum(frame, intensity)

    def __update_pos__(self, pos):
        """
        Update the position of each particle in the simulation
        """
        displacement = self.gen.normal(loc=0.0, scale=np.sqrt(2*self.diffusion_pixel * self.dt), size=(self.npart, 2))
        return pos + displacement
