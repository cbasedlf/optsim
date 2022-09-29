# -*- coding: utf-8 -*-
"""
Generating speckle patterns with (finite) lateral memory effect. 
From a point source, take a look at the speckle pattern at a given distance 
from the scattering layer when the source moves laterally.

Also implementation for ~infinite memory effect (illuminating the scattering
layer with plane waves with different tilts). In this case you can see the 
'warping' arround the borders (cirular behaviour) clearly

@author: Fernando
"""
#%% Import stuff
import numpy as np
import optsim as ops

#%% Define physical parameters
wvl = 532e-9 #wavelength
aperture_size = 50e-6 #physical size of the aperture
pxnum = 128 #number of pixels
pxsize = aperture_size / pxnum #pixel size
scat_sensor_distance = 300e-6 #distance between scattering layer and detector

#%% Generate source positions

xpos = np.squeeze(np.linspace(-20,20,20))*1e-6 #x positions
ypos = np.array((0,)) #y position
zpos = np.array((20e-6,)) #distance between source and scattering layer

# define total number of positions (object space)
num_pos = xpos.size*ypos.size*zpos.size
# Calculate array of 3D positions for the source
source_pos = np.zeros((num_pos,3)) #preallocating
if num_pos == 1:
    source_pos[0] = (xpos,ypos,zpos)
else:
    temp = 0
    for xidx in range(xpos.size):
        for yidx in range(ypos.size):
            for zidx in range(zpos.size):
                source_pos[temp,:] = (xpos[xidx],ypos[yidx],zpos[zidx])
                temp += 1
                pass
            pass
        pass
    pass

#%% Generate fields from point sources, propagate to the scattering layer, 
#and to the sensor plane

#Generate sources
print('generating sources...')
source = []
for idx in range(num_pos):
    source.append(ops.build_point(xpos = source_pos[idx,0],
                          ypos = source_pos[idx,1],
                          pxsize = pxnum,
                          delta = pxsize))
    pass
print('done')
source = np.asarray(source)

#propagate to scattering layer
print('propagating to scattering layer...')
field_before = []
for idx in range(num_pos):
    field_before.append(ops.rs_ang_spec_prop(Uin = source[idx,:,:], wvl = wvl,
                                delta = pxsize, z = zpos[0], padsize = 128))
    pass
print('done')
field_before = np.asarray(field_before)

#Generate a thin scattering medium
scat = ops.thin_scatter(size = field_before.shape[1]*pxsize, pxnum = field_before.shape[1], 
                        corr_width = 2*pxsize, strength = 2)

# Mask it (place a circular pupil)
mask, _, _ = ops.circAp(totalSize = field_before.shape[1], radius = 0.85) 
scat_layer = scat.thin_scat * mask
#Show random phase mask (scattering medium)
ops.show_field(scat_layer, mode = 'dark')

#Calculate field after scattering layer
field_after = field_before * scat_layer

#Propagate to sensor
print('propagating to sensor plane...')
field_sensor = []
for idx in range(num_pos):
    field_sensor.append(ops.rs_ang_spec_prop(Uin = field_after[idx,:,:],
                            wvl = wvl, delta = pxsize, 
                            z = scat_sensor_distance, padsize = 256))
    pass
print('done')
field_sensor = np.asarray(field_sensor)
field_sensor = np.moveaxis(field_sensor,0,2)

#show results at thesensor plane
anim1 = ops.show_vid(np.abs(field_sensor)**2,rate = 500, loop = True)

#%% Illuminate a scattering layer with plane waves with different tilts,
#look at the far field speckles
#Generate illuminations
print('generating illumination plane waves...')
illus = []
for idx in range(num_pos):
    #define ramp orientation
    ramp_angle = np.deg2rad(0)
    #define ramp strength (how many 2pi jumps in total)
    ramp_strength = 0.5*idx
    #build ramp image
    x = np.linspace(-1, 1, pxnum) #axes
    x,y = np.meshgrid(x,-x) #axes
    ramp = x*np.cos(ramp_angle) + y*np.sin(ramp_angle) #ramp profile
    #build complex mask
    ramp_mask = np.exp(1j*2*np.pi*ramp*ramp_strength)
    illus.append(ramp_mask)
print('done')
illus = np.asarray(illus)

#Generate a thin scattering medium
scat = ops.thin_scatter(size = pxnum*pxsize, pxnum = pxnum, 
                        corr_width = 2.5*pxsize, strength = 2)

# Mask it (place a circular pupil)
mask, _, _ = ops.circAp(totalSize = pxnum, radius = 0.85) 
scat_layer = scat.thin_scat * mask
#Show random phase mask (scattering medium)
ops.show_field(scat_layer, mode = 'dark')

#Calculate field after scattering layer
field_after_plane = illus * scat_layer

#Propagate to sensor
print('propagating to sensor plane...')
field_sensor_plane = []
for idx in range(num_pos):
    field_sensor_plane.append(ops.rs_ang_spec_prop(Uin = field_after_plane[idx,:,:],
                            wvl = wvl, delta = pxsize, 
                            z = 300e-6, padsize = 256))
    pass
print('done')
field_sensor_plane = np.asarray(field_sensor_plane)
field_sensor_plane = np.moveaxis(field_sensor_plane,0,2)

#show results at thesensor plane
anim2 = ops.show_vid(np.abs(field_sensor_plane)**2,rate = 500, loop = True)

