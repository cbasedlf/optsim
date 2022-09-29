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
Evaluates the Rayleigh-Sommerfeld diffraction integral for the propagation of a field between two planes. No paraxial approximation. This method was inspired by [fresbek_two_steps] and tries to tackle two main problems. First, when trying to propagate optical fields in microscopy simulations, you usually work with small fields of view (tens or hundreds of microns, typically). But, given that you work with wavelengths in the order of 400-800 nm (in the VIS), your mesh needs to be fine enough to sample these features. This entails that, for example, if you want to sample a 100 microns aperture, with pixel sizes in the order of 200 nm, you would need to use at least 500 pixels for representing your field. This is fine (computers are powerful nowadays), BUT if you want to propagate for long distances (lets say some milimiters or even a few centimeters), Fourier Transform propagation methods start introducing aliasing and the result looks terrible. The conventional solution to this is to do zero-padding
### lens_in_front_ft

# Quality of life
### show_img
### show_2img
### show_Nimg
### show_vid
### plot_scatter2d
### z2rgb
### show_field
### nparray2png
### save_images
### centralROI
### cropROI
### ROIclick

# General optics simulation tools
### circAp
### buildGauss
### buildSuperGauss
### filt_fourier
### gen_quadphase_paraxial
### gen_quadphase
### noisify
### corr2
### xcorr2d
### snr_img

# Light field / wavefront sensing
### cropFOV
### uncropFOV
### raw2centroid
### corrApert
### subap2lowres
### elementalCorrelations
### iseven
### isodd
### wave2zernike
### zernIndex
### zernikeRadialFunc
### zernike_nm
### zernike_noll
### zernikeArray
### grad2zernike
### zernike2wave
### gradient

# Neuronal activity simulation
### act_pulse
### temp_trace

# Miscellaneous
### cropSpeckle
### del_black_borders
### norm_dimension
