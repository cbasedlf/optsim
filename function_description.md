# Function descriptions
# Optical Propagations
## Generating fields arising from point sources
These functions, along with the optical propagation functions ahead, were inspired by the descriptions in the book:
Numerical Simulation of Optical Wave Propagation with Examples in MATLAB
    Author(s): Jason D. Schmidt
    https://spie.org/Publications/Book/866274?SSO=1
He made a wonderful work explaining the ideas, so I recommend going there for a detailed explanation behind them. In any case, all of them share some fundamentals. You need to input the number of pixels of your image, the position of the source (x,y), and the pixel size (the physical size of the mesh grid you want to simulate)
### build_point
Generates a field with a quadratic profile for both amplitud and phase.
### build_source
Generates a field with a Gaussian profile for both amplitud and phase.
### build_point_sinc
Generates a field with a sinc profile for both amplitud and phase.
### build_point_circ
Generates a field were the amplitude is just a circular (small) pupil and no phase (constant phase).
### build_delta
Generates a field were the amplitude is just a single pixel (like a delta).
### ft2
Discretized version of the Fourier Transform by using DFT
### ift2
Discretized version of the Inverse Fourier Transform by using DFT
### fraunhofer_prop
Evaluates the Fraunhoffer diffraction integral between two planes. Returns both the field and the mesh (grid spacing) at the output plane.
### fresnel_one_step
Evaluates the Fresnel diffraction integral between two planes. Propagation is done in a single step, which does not allow to choose the grid spacing at the output plane.
### fresnel_two_steps
Evaluates the Fresnel diffraction integral between two planes. Propagation is done in a two steps, which allows to choose the grid spacing at the output plane.
### ang_spec_prop
Evaluates the Fresnel diffraction integral between two planes using the angular spectrum method. As it solves Fresnel, this is assuming paraxial approximation.
### rs_ang_spec_prop
Evaluates the Rayleigh-Sommerfeld diffraction integral for the propagation of a field between two planes. No paraxial approximation. Does not take into account evanescent waves, so this method is nice if you want to propagate back and forth between two planes (you should get the exact same field when backgpropagating to the original plane).
### rs_ang_spec_prop_evanescent
Evaluates the Rayleigh-Sommerfeld diffraction integral for the propagation of a field between two planes. No paraxial approximation. Takes into account evanescent waves.
### rs_ang_spec_prop_multistep
Evaluates the Rayleigh-Sommerfeld diffraction integral for the propagation of a field between two planes. No paraxial approximation. This method was inspired by [fresbek_two_steps] and tries to tackle two main problems. First, when trying to propagate optical fields in microscopy simulations, you usually work with small fields of view (tens or hundreds of microns, typically). But, given that you work with wavelengths in the order of 400-800 nm (in the VIS), your mesh needs to be fine enough to sample these features. This entails that, for example, if you want to sample a 100 microns aperture, with pixel sizes in the order of 200 nm, you would need to use at least 500 pixels for representing your field. This is fine (computers are powerful nowadays), BUT if you want to propagate for long distances (lets say some milimiters or even a few centimeters), Fourier Transform propagation methods start introducing aliasing and the result looks terrible. The conventional solution to this is to do zero-padding, but for long distances the pad sizes become so big that even nowadays computers have memory troubles (you cannot propagate easily images with sizes on the order of tens of megapixels). This method tackles this by breaking the full propagation into several short distance propagations. After each propagation, the mesh can be recalculated (pixels become larger in physical size) and an absorbing window is applied in the borders of the field of view (so you do not need to do a huge zero-padding). The final result is quite good in the center of the field of view, but the border regions become a bit distorted (due to the absorbing window removing energy from the field after each propagation).
### lens_in_front_ft
Propagates a field from a plane in front of a lens, to the focal plane after the lens (so basically does a Fourier Transform with some scaling).
# Quality of life
### show_img
Plots a single input image (numpy 2D array)
### show_2img
Plots two images (numpy 2D array) side by side
### show_Nimg
Plots N images (numpy 2D array) in the same window
### show_vid
Shows an animation. Input should be a 3D numpy array, where the third axis corresponds to the frames (individual images)
### plot_scatter2d
Plots a 2D scatter plot. Input is a 2D numpy array, where each column corresponds to the X and Y coordinates of a single data point.
### z2rgb
Converts complex numbers to RGB color space, where the Saturation value corresponds to the intensity and the Hue and Value encode the phase. This is useful for repsenting intensity + phase distributions into a single image (instead of plotting amplitude and phase individually)
### show_field
Plots a complex field in a single image (using [z2rgb])
### nparray2png
Converts a 2D numpy array into a grayscale Image (PIL library) object, which can be easily exported (as a .png for example)
### save_images
Stores all the images from a 3D array into individual .png files. Useful for exporting batches of images.
### centralROI
Crops a Region of Interest, centered at the central part of an image.
### cropROI
Crops a Region of Interest, at any given position of an image.
### ROIclick
Crops a Region of Interest, with an arbitrary size and at any position of the image. Selection of the ROI is done by using OpenCV, by showing the image on screen and selecting the ROI with the mouse.
# General optics simulation tools
### circAp
Builds a circular aperture
### buildGauss
Builds a 2D Gaussian profile
### buildSuperGauss
Builds a 2D superGaussian profile
### filt_fourier
Filters an input image in the fourier domain with a given filter
### gen_quadphase_paraxial
Generates a quadratic phase profile (useful for simulating thin lenses in the paraxial approximation)
### gen_quadphase
Generates a quadratic phase profile
### noisify
Introduces noise into a signal, with a desired SNR
### corr2
Correlation coefficient between two images
### xcorr2d
Cross-correlation between two images
### snr_img
Calculates the SNR of an image

# Light field / wavefront sensing
### cropFOV
Crops an input image into N subimages. Useful for breaking an image taken after a microlens array into individual subimages from each lenslet.
### uncropFOV
Merges N subimages into an image
### raw2centroid
From a raw image taken through a microlens array, calculates the position of the centroid of each microimage.
### corrApert
Estudies correlations between microimages. From an input image, crops into subimages and correlates an individual image with all the others. Then returns the position of the correlation peaks.
### subap2lowres
Generates a low resolution image from a subaperture image.
### elementalCorrelations
### iseven
Checks if a number is even
### isodd
Checks if a number is odd
### wave2zernike
Decomposes an input wavefront into the basis of zernike polynomials
### zernIndex
Calculates the [n,m] index from the Noll index [j]
### zernikeRadialFunc
Calculates the Zernike radial function of order [n,m]
### zernike_nm
Calculates the Zernike polynomial with radial index [n] and azimuthal index [m]
### zernike_noll
Generates teh zernike polynomial with Noll index [j]
### zernikeArray
Generates an array of Zernike polynomials (using a list of Noll indexes)
### grad2zernike
From the partial X,Y derivatives of a wavefront, calculate the zernike decomposition of the original wavefront.
### zernike2wave
Calculate the phase profile of a wavefront from its zernike decomposition coefficients.
### gradient
Calculate the 2D gradient of an image

# Neuronal activity simulation
### act_pulse
Generates an activation pulse for a neuron. Linear ramp for the rise and an exponential fall. You can tune the ratio between rise/fall, and the exponential decay rate.
### temp_trace
Generate a temporal activity trace with many activation pulses.

# Miscellaneous
### cropSpeckle
Crops the central region of a speckle pattern
### del_black_borders
Removes black borders on the sides of an image
### norm_dimension
Normalizes the entries of a matrix. Can be used to normalize individual rows/columns independently
