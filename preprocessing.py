#!/usr/bin/env python

### Author: David C. Heson. April 2023

### Imports

import warnings
warnings.filterwarnings('ignore') # the photutils functions I am currently using are being
                                  # deprecated so they spit out a long warning when imported
                                  # I will change the code to the reccomended functions later

import astropy
from pathlib import Path
from astropy.nddata import CCDData
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from photutils import detect_sources, detect_threshold, Background2D, MedianBackground
import warnings

warnings.resetwarnings()

### Functions to be used

def dark_bias_hdr(directory):

    os.makedirs(directory + "/dark_bias", exist_ok = True)

    for image in os.listdir(directory):

        if image == "dark_bias":
            continue

        # Load image
        hdul = fits.open(directory + "/" + image)
        data = hdul[0].data.astype(float)

        # Load bias info from header
        detector_bias = hdul[0].header['DETBIAS']
        detector_gain = hdul[0].header['GAIN']
        q1_offset = hdul[0].header['Q1OFFSET']
        q2_offset = hdul[0].header['Q2OFFSET']
        q3_offset = hdul[0].header['Q3OFFSET']
        q4_offset = hdul[0].header['Q4OFFSET']
        naxis1 = hdul[0].header['NAXIS1']
        naxis2 = hdul[0].header['NAXIS2']

        # Subtract detector bias and apply quadrant preamp offset correction
        for i in range(4):
            y_start = i * (naxis2 // 4)
            y_end = (i + 1) * (naxis2 // 4)

            for j in range(4):
                x_start = j * (naxis1 // 4)
                x_end = (j + 1) * (naxis1 // 4)

                quadrant = data[y_start:y_end, x_start:x_end]
                quadrant -= q1_offset
                quadrant -= q2_offset
                quadrant -= q3_offset
                quadrant -= q4_offset
                quadrant -= detector_bias
                quadrant /= detector_gain

                data[y_start:y_end, x_start:x_end] = quadrant

        # Save processed image
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(directory + "/dark_bias/" + image, overwrite=True)
        hdul.close()

def dark_bias_hdr_noqdr(directory):

    os.makedirs(directory + "/dark_bias", exist_ok=True)

    for image in os.listdir(directory):

        if image == "dark_bias":
            continue

        # Load image
        hdul = fits.open(directory + "/" + image)
        data = hdul[0].data.astype(float)

        # Load bias info from header
        detector_bias = hdul[0].header['DETBIAS']
        detector_gain = hdul[0].header['GAIN']

        # Subtract detector bias
        data -= detector_bias
        data /= detector_gain

        # Save processed image
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(directory + "/dark_bias/" + image, overwrite=True)
        hdul.close()

def im_median_combine(directory):

    frames = []

    for image in os.listdir(directory):

        hdul = fits.open(directory + "/" + image)
        frames.append(hdul[0].data.astype(float))
        hdul.close()

        # Combine flat frames
        combined = np.median(frames, axis=0)

        # Save combined flat frame
        hdu = fits.PrimaryHDU(combined)
        hdu.writeto(directory + '/combined.fits', overwrite=True)
        hdul.close()

def im_sigmaclip_combine(directory, sigma=3):

    frames = []

    for image in os.listdir(directory):

        if image == "combined.fits":
            continue

        hdul = fits.open(directory + "/" + image)
        frames.append(hdul[0].data.astype(float))
        hdul.close()

    # Calculate median and standard deviation of the frames
    median = np.median(frames, axis=0)
    std = np.std(frames, axis=0)

    # Sigma clipping
    clipped_frames = []
    for frame in frames:
        mask = np.abs(frame - median) < sigma * std
        clipped_frames.append(frame[mask])

    # Combine the clipped frames
    combined = np.median(clipped_frames, axis=0)

    # Save combined frame
    hdu = fits.PrimaryHDU(combined)
    hdu.writeto(directory + '/combined.fits', overwrite=True)
    hdul.close()

def flat_division(flat, directory):

    flat_hdul = fits.open(flat)
    flat_data = flat_hdul[0].data.astype(float)
    os.makedirs(directory + "/flattened", exist_ok = True)

    for image in os.listdir(directory):

        if image == "flattened":
            continue

        hdul = fits.open(directory + "/" + image)
        data = hdul[0].data.astype(float)

        data /= flat_data

        hdu = fits.PrimaryHDU(data)
        hdu.writeto(directory + "/flattened/" + image, overwrite=True)

    flat_hdul.close()

def im_substraction(directory, combined):

    combined_hdul = fits.open(combined)
    combined_data = combined_hdul[0].data.astype(float)
    os.makedirs(directory + "/substracted", exist_ok = True)

    for image in os.listdir(directory):

        if image == "substracted":
            continue

        if image == "flattened":
            continue

        hdul = fits.open(directory + "/" + image)
        data = hdul[0].data.astype(float)

        data -= combined_data

        hdu = fits.PrimaryHDU(data)
        hdu.writeto(directory + "/substracted/" + image, overwrite=True)
        hdul.close()

    combined_hdul.close()

### Script itself

main_dir = input("Please enter the location of the data directory: ")

science_im = main_dir + "/science"
sky_im = main_dir + "/sky"
flat_im = main_dir + "/flat"

#dark_bias_hdr(science_im)
#dark_bias_hdr(sky_im)
#dark_bias_hdr(flat_im)

dark_bias_hdr_noqdr(science_im)
dark_bias_hdr_noqdr(sky_im)
dark_bias_hdr_noqdr(flat_im)

science_im = science_im + "/dark_bias"
sky_im = sky_im + "/dark_bias"
flat_im = flat_im + "/dark_bias"

im_median_combine(flat_im)
flat_im = flat_im + "/combined.fits"

flat_division(flat_im, science_im)
flat_division(flat_im, sky_im)

science_im = science_im + "/flattened"
sky_im = sky_im + "/flattened"

im_median_combine(sky_im)
sky_im = sky_im + "/combined.fits"

im_substraction(science_im, sky_im)
science_im = science_im + "/substracted"

im_median_combine(science_im)

print("Combined calibrated science image created at: " + science_im + "combined.fits")
