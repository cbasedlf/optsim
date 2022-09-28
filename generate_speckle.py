# -*- coding: utf-8 -*-
"""
Generating speckle patterns in different configurations.

All the methods are based on generating a phase mask that will act as a thin
scattering layer (so, infinite memory effect). After that, we propagate the
field a given distance to generate the speckle pattern. If you want to simulate
near field / microscopy, you can use propagation methods such as Fresnel or 
the angular spectral method. If you do not care about distances/sizes, the 
simplest is to asume far field and just use a Fourier Transform.

@author: Fernando
"""
#%% Import stuff
import numpy as np
import optsim as ops

#%% Define physical parameters
wvl = 532e-9 #wavelength
aperture_size = 100e-6 #physical size of the aperture
pxnum = 64 #number of pixels
pxsize = aperture_size / pxnum #pixel size
#%% Generate a thin scattering medium
scat = ops.thin_scatter(size = aperture_size, pxnum = pxnum, 
                        corr_width = 4e-6, strength = 2)

#%% Mask it (place a circular pupil)

mask, _, _ = ops.circAp(totalSize = pxnum, radius = 0.85) 

scat_layer = scat.thin_scat * mask
#Show random phase mask (scattering medium)
ops.show_field(scat_layer, mode = 'dark')

#%% Propagate a field through the scattering medium to generate speckle
#Short distance propagation using the angular spectral method:
prop_field = ops.rs_ang_spec_prop(Uin = scat_layer, wvl = wvl, 
                    delta = pxsize, z = 2e-3, padsize = 128)
#Show field after propagation (speckle)
ops.show_2img(np.abs(prop_field), np.angle(prop_field)) #show amplitud and phase
ops.show_img(np.abs(prop_field)**2) #show intensity

#Far field propagation (Fourier transform):
speckle = ops.ft2(np.pad(scat_layer, [128,128], mode = 'constant'), 1)
ops.show_2img(np.abs(speckle), np.angle(speckle))#show amplitud and phase
ops.show_img(np.abs(speckle)**2)#show intensity

#%% Generate speckle and see it evolving along z axis
z_range = np.linspace(0, 1e-3, 100)#define propagation distances
speckle_evolving = [] #initialize

#Calculate speckle patterns at different distances from the aperture
for zidx in range(z_range.size):
    speckle_evolving.append(np.abs(ops.rs_ang_spec_prop(Uin = scat_layer, wvl = wvl, 
                        delta = pxsize, z = z_range[zidx], padsize = 128))**2)

#convert to numpy form, rearrange
speckle_evolving = np.asarray(speckle_evolving)
speckle_evolving = np.moveaxis(speckle_evolving,0,2)
#show animation
anim = ops.show_vid(speckle_evolving, rate = 100, 
                        fig_size = (5,3), loop = True)

