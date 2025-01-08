import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import imageio
from datetime import datetime

# Readme!
# This Code is originally written by Aikaterina Grava and has ben modified to be used in automatic error analysis


# Function to create a fluorescent particle with a Gaussian intensity profile
def create_fluorescent_particle(center, std_dev, intensity=255, size=(200, 200)):
    x, y = np.ogrid[:size[0], :size[1]]  # grid of coordinates
    distance_squared = (x - center[0])**2 + (y - center[1])**2
    gaussian_spot = intensity * np.exp(-distance_squared / (2 * std_dev**2))

    # Alan: ensure that always the same total intensity is used, TODO check if this is correct
    # tot_intensity = np.sum(gaussian_spot)
    # gaussian_spot = gaussian_spot / tot_intensity * 2500 # 10000 is the total intensity chosen arbitrary

    gaussian_spot = np.clip(gaussian_spot, 0, 255).astype(np.uint8)  # Ensure values are valid
    return gaussian_spot


# Function to add grayscale noise (background)
def add_noise(image, noise_density=0.1):
    noisy_image = image.copy()
    num_pixels = noisy_image.size
    num_noisy_pixels = int(noise_density * num_pixels)

    # Generate random coordinates for noisy pixels
    noise_coords = [np.random.randint(0, i - 1, num_noisy_pixels) for i in noisy_image.shape]

    # Assign random intensity values (grayscale noise)
    noisy_intensities = np.random.randint(0, 256, num_noisy_pixels)  # Random values between 0 and 255
    noisy_image[noise_coords[0], noise_coords[1]] = noisy_intensities

    return noisy_image


# Function to calculate the displacement due to Brownian motion
def brownian_displacement(D, time_step):
    return np.random.normal(loc=0.0, scale=np.sqrt(2 * D * time_step), size=2)  # x and y displacement


# Function to update the position of the particle in each frame
def update(frame_num, current_pos, D, time_step, image_size, radius, std_dev):
    # Calculate Brownian displacement
    displacement = brownian_displacement(D, time_step)
    current_pos += displacement

    # Ensure the particle stays within bounds
    current_pos[0] = np.clip(current_pos[0], radius, image_size[0] - radius)
    current_pos[1] = np.clip(current_pos[1], radius, image_size[1] - radius)
    current_pos = list(current_pos)

    # Create the new frame with the particle at the updated position
    frame = create_fluorescent_particle(current_pos, std_dev, size=image_size)
    #frame = add_noise(frame)

    return frame, current_pos


# Function to start the simulation
def run_simulation(fps, pixel_size, output_folder, temperature=293,
                   viscosity=0.001, num_frames = 900, image_size=(512, 512),
                   particle_radius=250):
    """
    Run adapted Aikaterina simulation

    param fps: int frames per second
    param pixel_size: pixel size in nm
    param output_folder: output folder
    param temperature: temperature in K, default is 293 K
    param viscosity: viscosity in Pa.s, default is 0.001 Pa.s
    param num_frames: number of frames, default is 900
    param image_size: image size in pixels, default is (512, 512)
    param particle_radius: particle radius in nm, default is 250 nm
    """
    temperature

    # Define particle properties
    particle_radius_m = 0.5e-6  # 0.5 μm in meters

    # Calculate the diffusion coefficient (Stokes-Einstein equation)
    k_B = 1.38e-23  # Boltzmann constant (J/K)
    D = k_B * temperature / (6 * np.pi * viscosity * particle_radius_m)  # m²/s

    # Convert diffusion coefficient to pixel space
    D_pixel = D * (1e9 / pixel_size)**2  # nm²/s to pixels²/s

    # Simulation parameters
    time_step = 1 / fps  # Time per frame in seconds
    std_dev = particle_radius/pixel_size  # chosen to be the particle radius
    noise_density = 0.1  # Background noise density

    # Initial position set to the center of the image
    current_pos = [image_size[0] // 2, image_size[1] // 2]
    pos_history = []

    # Save images to the output folder
    def save_frame(frame_num, frame):
        frame = frame.astype(np.uint8)
        imageio.imsave(os.path.join(output_folder, f"frame_{frame_num + 1:03d}.tif"), frame.T)

    frame = create_fluorescent_particle(current_pos, std_dev, size=image_size)
    # check if there are alredy frames in the folder, if so delete all
    for file in os.listdir(output_folder):
        if file.endswith('.tif'):
            os.remove(os.path.join(output_folder, file))
    for i in range(num_frames):
        save_frame(i, frame)
        pos_history.append(current_pos)
        frame, current_pos = update(i, current_pos, D_pixel, time_step, image_size, int(std_dev), std_dev)

    if os.path.exists(output_folder + '/pos_history.npy'):
        os.remove(output_folder + '/pos_history.npy')
    np.save(output_folder + '/pos_history.npy', np.array(pos_history))
    return
