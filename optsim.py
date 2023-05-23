"""
Collection of functions useful for general optical propagation simulations.
Also generating speckle, and some light field / wavefront sensing routines.
***************************************************************************
cbasedlf/optsim is licensed under the
                MIT License
***************************************************************************
If you found it useful, please cite the repo in your projects.
F. Soldevila (@cbasedlf on Github). November 10th, 2022.
"""

#%% General use
import numpy as np
from numpy.fft import fft as fft, fft2 as fft2
from numpy.fft import ifft as ifft, ifft2 as ifft2
from numpy.fft import fftshift as fftshift, ifftshift as ifftshift
import scipy as sc
import cv2

#plotting
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Classes
'''
Honestly I do not know why I defined the scattering layer as a class. It was 
probably my first week learning python and I just wanted to try. You can define
a function that returns a matrix with [thin_scat] and forget about the rest I
guess.
'''

class thin_scatter():
    def __init__(self, size, pxnum, corr_width, strength):
        '''
        [corr_width] controls the 'correlation width' of the scattering layer.
        The lower, the bigger the detail of the layer. IN LENGTH UNITS
        
        [strength] controls the amount of balistic photons that go through the 
        layer (controls if the phase jumps introduced by the thin layer are
        bigger or smaller than one wavelength)
        '''
        delta = size / pxnum #mesh spacing (spatial domain)
        delta_f = 1 / (pxnum*delta) #mesh spacing (frequency domain)
        corr_width_px = int(corr_width / delta) #calculate corr_width in pixels
        #Build gaussian in spatial domain, with the desired width
        lowpass = buildGauss(px = pxnum,sigma = (corr_width_px,corr_width_px),
                             center = (int(pxnum/2),int(pxnum/2)),
                             phi = 0)
        #Convert to frequency domain (to do the filtering)
        lowpass_ft = ft2(lowpass, delta)
        self.size = size
        self.pxnum = pxnum
        self.delta = delta
        self.seed = np.random.rand(self.pxnum,self.pxnum)
        #Filter in frequency domain so detail size corresponds to corr_width
        self.phase = np.real(ift2(ft2(self.seed,delta)*lowpass_ft,delta_f))
        #shift between -0.5 and 0.5
        self.phase = (self.phase - np.min(self.phase)) / (np.max(self.phase) - np.min(self.phase)) - 0.5
        #build final phase mask (no wrap here)
        self.phase = self.phase*2*np.pi*strength
        #build 'field' (wrapped phase)
        self.thin_scat = np.exp(1j*self.phase)
        pass
    pass

#%% Optical propagation functions (and related)
''' 
    Derived from:
    Numerical Simulation of Optical Wave Propagation with Examples in MATLAB
    Author(s): Jason D. Schmidt
    https://spie.org/Publications/Book/866274?SSO=1
'''
###############################################################################
####### For generating fields that arise from 'point' sources #################
###############################################################################
def build_point(xpos, ypos, pxsize, delta):
    '''
    build_point generates the field of a point source at the plane of the 
    point source. Useful for propagation simulations

    Parameters
    ----------
    xpos, ypos : position of the source (length units)
    pxsize : number of pixels of the grid
    delta : grid spacing at the source plane (length units)

    Returns
    -------
    source_field : field of the point source

    '''
    x = np.arange(-pxsize/2, pxsize/2, 1)*delta
    X,Y = np.meshgrid(x,-x)
    exp1 = np.exp(-np.sqrt((X-xpos)**2 + (Y-ypos)**2)**2 / (2*delta**2))
    exp2 = np.exp(-1j*np.sqrt((X-xpos)**2 + (Y-ypos)**2)**2 / (2*delta**2))
    source_field = exp1 * exp2
    return source_field

def build_source(xpos, ypos, pxsize, delta, source_size):
    '''
    build_source generates the field of a point source at the plane of the 
    point source. Useful for propagation simulations

    Parameters
    ----------
    xpos, ypos : position of the source (length units)
    pxsize : number of pixels of the grid
    delta : grid spacing at the source plane (length units)
    source_size : size of the source (length units). Needs to be bigger than
                    delta

    Returns
    -------
    source_field : field of the point source

    '''
    gauss_width = source_size/4
    x = np.arange(-pxsize/2, pxsize/2, 1)*delta
    X,Y = np.meshgrid(x,-x)
    exp1 = np.exp(-np.sqrt((X-xpos)**2 + (Y-ypos)**2)**2 / (2*gauss_width**2))
    exp2 = np.exp(-1j*np.sqrt((X-xpos)**2 + (Y-ypos)**2)**2 / (2*gauss_width**2))
    source_field = exp1 * exp2
    return source_field

def build_point_sinc(xpos, ypos, N, apsize, wvl, z):
    '''
    build_point_sinc generates the field of a point source 
    at the plane of the point source. Useful for propagation simulations.
    Uses a different model than build_point (here we model the source
    as a sinc function)
    Parameters
    ----------
    xpos, ypos : position of the source (length units)
    N : number of grid points
    apsize : physical size of the aperture at the observation plane
            (length units)
    wvl : wavelength of the source (length units)
    z : propagation distance (length units)
    Returns
    -------
    source_field : field of the point source
    delta : grid spacing at the propagated plane (length units)
    '''
    k = 2*np.pi / wvl
    arg = apsize / (wvl*z)
    delta = 1 / (100*arg)
    x = np.arange(-N/2, N/2, 1) * delta
    x1,y1 = np.meshgrid(x,-x)
    
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return phi, rho
    theta1, r1 = cart2pol(x1, y1)
    thetapos, rpos = cart2pol(xpos,ypos)
    
    A = wvl*apsize #sets field amplitude to 1 in obs plane
    
    E1 = np.exp(-1j*k/(2*z)*np.sqrt((x1-xpos)**2+(y1-ypos)**2)**2)
    E2 = 1
    source_field = A*E1*E2*arg**2*np.sinc(arg*(xpos-x1))*np.sinc(arg*(ypos-y1))

    return source_field, delta

def build_point_circ(xpos, ypos, r, N, apsize):
    '''
    build_point_circ generates a point source with the simplest model:
        just a circular aperture as the irradiance, no phase

    Parameters
    ----------
    xpos, ypos : position of the source
    r : radius of the source
    N : number of grid points
    apsize : size of the grid
    Returns
    -------
    source_field : amplitude mask representing the source
    '''
    delta = apsize/N #grid spacing
    #Generate grid
    x = np.arange(-N/2, N/2, 1)*delta
    X,Y = np.meshgrid(x,-x)
    #Generate radius (polar)
    rho = np.abs((X-xpos) + 1j*(Y-ypos))
    #Generate mask with circular aperture of given size, centered at source pos
    mask = rho<r
    #Build the source
    source_field = np.ones((N,N)) * mask
    return source_field

def build_delta(xpos, ypos, pxsize, delta):
    '''
    build_delta generates a point source that is a delta function
    (a single pixel)

    Parameters
    ----------
    xpos, ypos : position of the source
    pxsize : number of pixels of the grid
    delta : grid spacing at the source plane (length units)
    Returns
    -------
    source_field : amplitude mask representing the source
    '''
    #Generate grid
    x = np.arange(-pxsize/2, pxsize/2, 1)*delta
    X,Y = np.meshgrid(x,-x)
    #find position of the source (in pixel index, finds the closest one)
    xidx = np.abs(x-xpos).argmin()
    yidx = np.abs(-x-ypos).argmin()
    #Generate source amplitude (all zeros)
    source_field = np.zeros((pxsize,pxsize))
    #Build the source
    source_field[yidx,xidx] = 1
    return source_field
###############################################################################
###Optical propagation routines (Fresnel, Fraunhoffer, Rayleigh-Sommerfeld)####
###############################################################################
def ft(g, delta):
    '''
    ft performs a discretized version of a Fourier Transform by using DFT

    Parameters
    ----------
    g : input signal (sampled discretely) on the spatial(temporal) domain
    delta : grid spacing spatial(temporal) domain. length(time) units

    Returns
    -------
    G : Fourier Transform

    '''
    G = fftshift(fft(ifftshift(g)))*delta
    return G

def ift(G, delta_f):
    '''
    ift performs a discretized version of an Inverse Fourier Transform
    by using DFT

    Parameters
    ----------
    G : input signal (sampled discretely) on the frequency domain
    delta_f : grid spacing frequency domain. 1/length(1/time) units

    Returns
    -------
    g : Inverse Fourier Transform

    '''
    n = G.shape[0]
    g = ifftshift(ifft(fftshift(G)))*(n*delta_f)
    return g

def ft2(g, delta):
    '''
    ft2 performs a discretized version of a Fourier Transform by using DFT

    Parameters
    ----------
    g : input field (sampled discretely) on the spatial domain
    delta : grid spacing spatial domain (length units)

    Returns
    -------
    G : Fourier Transform

    '''
    G = fftshift(fft2(ifftshift(g)))*delta**2
    return G

def ift2(G, delta_f):
    '''
    ift2 performs a discretized version of an Inverse Fourier Transform
    by using DFT

    Parameters
    ----------
    G : input field (sampled discretely) on the frequency domain
    delta_f : grid spacing frequency domain (1/length units)

    Returns
    -------
    g : Inverse Fourier Transform

    '''
    n = G.shape[0]
    g = ifftshift(ifft2(fftshift(G)))*(n*delta_f)**2
    return g

def fraunhofer_prop(Uin, wvl, delta, z, padsize = False):
    '''
    franhofer_prop evaluates the Fraunhofer diffraction integral for an 
    optical field between two planes.

    Parameters
    ----------
    Uin : Input field
    wvl : wavelength of the field
    delta : grid spacing on the input plane (spatial domain)
    z : propagation distance
    padsize : size of the padding (if wanted)

    Returns
    -------
    Uout : Output field after propagation
    x2 : x-grid on the output plane
    y2 : y-grid on the output plane

    '''
    N = Uin.shape[0]
    #Padding (if specified)
    if padsize != False:
        #pad the input
        Uin = np.pad(Uin, ((padsize,padsize),(padsize,padsize)), 'constant')
        #Calculate new size in pixels
        N = Uin.shape[0]
        pass
    k = 2*np.pi/wvl #wavenumber
    #spatial frequencies at source plane
    delta_f = 1/(N*delta)
    f_x1 = np.arange(-N/2, N/2, step = 1)*delta_f #frequency axis
    #generate observation plane coordinates
    x2, y2 = np.meshgrid(f_x1*z*wvl,-f_x1*z*wvl)
    #Calculate field at the output
    Uout = 1/(1j*wvl*z)*np.exp(1j*k/(2*z)*(x2**2+y2**2))*ft2(Uin,delta)
    
    return Uout, x2, y2

def fresnel_one_step(Uin,wvl,delta,z,padsize=False):
    '''
    fresnel_one_step evaluates the Fresel diffraction integral for an 
    optical field between two planes. It does it in a single step, which 
    does not allow for controlling the grid spacing at the output (observation)
    plane

    Parameters
    ----------
    Uin : Input field
    wvl : wavelength of the field
    delta : grid spacing on the input plane (spatial domain)
    z : propagation distance
    padsize : size of the padding (if wanted)

    Returns
    -------
    Uout : Output field after propagation
    x2 : x-grid on the output plane
    y2 : y-grid on the output plane

    '''
    N = Uin.shape[0]
    #Padding (if specified)
    if padsize != False:
        #pad the input
        Uin = np.pad(Uin, ((padsize,padsize),(padsize,padsize)), 'constant')
        #Calculate new size in pixels
        N = Uin.shape[0]
        pass
    k = 2*np.pi/wvl
    #Build source-plane coords
    x = np.arange(-N/2, N/2, step = 1)*delta #x-axis
    x1,y1 = np.meshgrid(x,-x) #generate grid
    #Calculate output plane coords (delta_f scaled by the geometry wvl*z)
    x2 = np.arange(-N/2, N/2, step = 1)*wvl*z/(N*delta)
    x2,y2 = np.meshgrid(x2,-x2)
    #evaluate the Fresnel-Kirchhoff integral
    Uout = 1/(1j*wvl*z)*np.exp(1j*k/(2*z)*(x2**2+y2**2))*ft2(Uin*np.exp(1j*k/(2*z)*(x1**2+y1**2)),delta)
    return Uout,x2,y2

def fresnel_two_steps(Uin,wvl,delta1,delta2,z,padsize=False):
    '''
    fresnel_two_steps evaluates the Fresel diffraction integral for an 
    optical field between two planes. It does it in two steps, which 
    allows for controlling the grid spacing at the output (observation)
    plane
    
    Parameters
    ----------
    Uin : Input field
    wvl : wavelength of the field
    delta1 : grid spacing on the input plane (spatial domain)
    delta2 : grid spacing on the output plane (spatial domain)
    z : propagation distance
    padsize : size of the padding (if wanted)

    Returns
    -------
    Uout : Output field (observation plane) after propagation
    x2 : x-grid on the output plane
    y2 : y-grid on the output plane
    '''
    N = Uin.shape[0] #Number of gridpoints
    #Padding (if specified)
    if padsize != False:
        #pad the input
        Uin = np.pad(Uin, ((padsize,padsize),(padsize,padsize)), 'constant')
        #Calculate new size in pixels
        N = Uin.shape[0]
        pass
    k = 2*np.pi/wvl #wavenumber
    #magnification
    m = delta2/delta1
    #Build source-plane coords
    x = np.arange(-N/2, N/2, step = 1)*delta1 #x-axis
    x1, y1 = np.meshgrid(x,-x) #generate grid
    
    ### Popagate to intermediate plane ###
    z1 = z/(1-m) #intermediate propagation distance
    #grid spacing on the auxiliary plane
    delta_aux = wvl*np.abs(z1)/(N*delta1) 
    #Build source-plane coords
    x_aux = np.arange(-N/2,N/2,step=1)*delta_aux #x-axis
    x_aux, y_aux = np.meshgrid(x_aux,-x_aux) #generate auxiliary grid
    #Evaluate Fresnel-Kirchoff integral
    Uaux = 1/(1j*wvl*z1)*np.exp(1j*k/(2*z1)*(x_aux**2+y_aux**2))*ft2(Uin*np.exp(1j*k/(2*z1)*(x1**2+y1**2)),delta1)
    ### Propagate to observation plane ###
    z2 = z - z1
    #Build source-plane coords
    x2 = np.arange(-N/2,N/2,step=1)*delta2 #x-axis
    x2, y2 = np.meshgrid(x2,-x2) #generate observation plane grid
    #Evaluate the Fresnel-Kirchhoff integral
    Uout = 1/(1j*wvl*z2)*np.exp(1j*k/(2*z2)*(x2**2+y2**2))*ft2(Uaux*np.exp(1j*k/(2*z2)*(x_aux**2+y_aux**2)),delta_aux)
    return Uout,x2,y2

def ang_spec_prop(Uin, wvl, delta1, delta2, z, padsize = False):
    '''
    ang_spec_prop evaluates the Fresel diffraction integral for an 
    optical field between two planes using the angular-spectrum method 
    Assumes paraxial approximation!

    Parameters
    ----------
    Uin : Input field (source plane)
    wvl : wavelength of the field
    delta1 : grid spacing on the input plane (spatial domain)
    delta2 : grid spacing on the output plane (spatial domain)
    z : propagation distance
    padsize : size of the padding (if wanted). False by default

    Returns
    -------
    Uout : Output field (observation plane)
    x2 : x-grid on the output plane
    y2 : y-grid on the output plane

    '''
    N = Uin.shape[0] #number of pixels (assume square grid)
    #Padding (if specified)
    if padsize != False:
        #pad the input
        Uin = np.pad(Uin, ((padsize,padsize),(padsize,padsize)), 'constant')
        #Calculate new size in pixels
        N = Uin.shape[0]
        pass
    k = 2*np.pi/wvl #wavenumber
    #Build source-plane coords
    x = np.arange(-N/2, N/2, step = 1)*delta1 #x-axis
    x1, y1 = np.meshgrid(x,-x) #generate grid
    r1sq = x1**2 + y1**2
    #Spatial frequencies at source plane
    delta_f1 = 1/(N*delta1)
    f_x1 = np.arange(-N/2, N/2, step = 1)*delta_f1 #frequency axis
    #Generate mesh
    f_x1, f_y1 = np.meshgrid(f_x1,-f_x1)
    fsq = f_x1**2 + f_y1**2
    #Scaling parameter
    m = delta2/delta1
    #Spatial grid at observation plane
    x = np.arange(-N/2, N/2, step = 1)*delta2 #x-axis
    x2, y2 = np.meshgrid(x,-x) #generate grid
    r2sq = x2**2 + y2**2
    #Quadratic phase factors
    Q1 = np.exp(1j*k/2*(1-m)/z*r1sq)
    Q2 = np.exp(-1j*2*np.pi**2*z/m/k*fsq)
    Q3 = np.exp(1j*k/2*(m-1)/(m*z)*r2sq)
    #Calculate propagated field
    Uout = Q3*ift2(Q2*ft2(Q1*Uin/m,delta1),delta_f1)
    
    return Uout, x2, y2

def rs_ang_spec_prop(Uin, wvl, delta, z, padsize = False):
    '''
    rs_ang_spec_prop evaluates the diffraction integral for an 
    optical field between two planes using the angular-spectrum method.
    Does not use paraxial approximation (equivalent to Rayleigh-Sommerfeld
    theory).
    Does not take into account evanescent waves (so, you can propagate back
    and forth between two planes and you will get the same field)

    Parameters
    ----------
    Uin : Input field (source plane)
    wvl : wavelength of the field
    delta : grid spacing on the input plane (spatial domain)
    z : propagation distance
    padsize : size of the padding (if wanted). False by default

    Returns
    -------
    Uout : Output field (observation plane)
    '''
    N = Uin.shape[0] #number of pixels (assume square grid)
    #Padding (if specified)
    if padsize != False:
        #pad the input
        Uin = np.pad(Uin, ((padsize,padsize),(padsize,padsize)), 'constant')
        #Calculate new size in pixels
        N = Uin.shape[0]
        pass
    k = 2*np.pi/wvl #wavenumber
    #Build source-plane coords
    x = np.arange(-N/2, N/2, step = 1)*delta #x-axis
    x1, y1 = np.meshgrid(x,-x) #generate grid
    #Spatial frequencies at source plane
    delta_f = 1/(N*delta)
    f_x = np.arange(-N/2, N/2, step = 1)*delta_f #frequency axis
    #Generate mesh
    f_x, f_y = np.meshgrid(f_x,-f_x)
    #Transfer function
    #Term inside the square root
    sr = 1 - (wvl*f_x)**2 - (wvl*f_y)**2
    #see where the factor inside the square root is positive. Travelling waves
    sr_prop = sr*(sr > 0) 
    #Calculate transfer functions
    H = np.exp(1j*k*z*np.sqrt(sr_prop)) #for traveling waves
    #apply frequency cut (only where the frequency is lower 
    #than the frequency cut of the system)
    H = H*(sr > 0) 
    #Calculate propagated field
    Uout = ift2(ft2(Uin,delta)*H, delta_f)
    
    return Uout

def rs_ang_spec_prop_evanescent(Uin, wvl, delta, z, padsize = False):
    '''
    rs_ang_spec_prop_evanescent evaluates the diffraction integral for an 
    optical field between two planes using the angular-spectrum method.
    Does not use paraxial approximation (equivalent to Rayleigh-Sommerfeld
    theory)
    Takes into account evanescent waves (so, you cannot go back and forth 
    between two planes with the same result)

    Parameters
    ----------
    Uin : Input field (source plane)
    wvl : wavelength of the field
    delta : grid spacing on the input plane (spatial domain)
    z : propagation distance
    padsize : size of the padding (if wanted). False by default

    Returns
    -------
    Uout : Output field (observation plane)
    '''
    N = Uin.shape[0] #number of pixels (assume square grid)
    #Padding (if specified)
    if padsize != False:
        #pad the input
        Uin = np.pad(Uin, ((padsize,padsize),(padsize,padsize)), 'constant')
        #Calculate new size in pixels
        N = Uin.shape[0]
        pass
    k = 2*np.pi/wvl #wavenumber
    #Build source-plane coords
    x = np.arange(-N/2, N/2, step = 1)*delta #x-axis
    x1, y1 = np.meshgrid(x,-x) #generate grid
    #Spatial frequencies at source plane
    delta_f = 1/(N*delta)
    f_x = np.arange(-N/2, N/2, step = 1)*delta_f #frequency axis
    #Generate mesh
    f_x, f_y = np.meshgrid(f_x,-f_x)
    #Transfer function
    #Term inside the square root
    sr = 1 - (wvl*f_x)**2 - (wvl*f_y)**2
    #see where the factor inside the square root is positive. Travelling waves
    sr_prop = sr*(sr>0) 
    #see where the factor inside the square root is negative. Evanescent waves
    sr_evanescent = sr * (sr < 0)
    #Calculate transfer functions
    H_prop = np.exp(1j*k*z*np.sqrt(sr_prop)) #for traveling waves
    #apply frequency cut (only where the frequency is lower than 
    #the frequency cut of the system)
    H_prop = H_prop * (sr > 0) 
    H_evanescent = np.exp(-k*z*np.sqrt(-sr_evanescent)) #For evanescent waves
    H_evanescent = H_evanescent*(sr < 0)
    #Combine into full transfer function
    H = H_prop + H_evanescent
    #Calculate propagated field
    Uout = ift2(ft2(Uin,delta)*H, delta_f)
    return Uout

def rs_ang_spec_prop_multistep(Uin, wvl, delta, z, zsteps,
                                padsize = False, resample_period = False):
    '''
    rs_ang_spec_prop_multistep evaluates the diffraction integral for an 
    optical field between two planes using the angular-spectrum method.
    Does not use paraxial approximation (equivalent to Rayleigh-Sommerfeld
    theory).
    Does the propagation in N steps, in order to avoid aliasing.
    After every propagation, uses absorbing boundaries
    It can resample the field after resample_period propagations, thus
    increasing the mesh spacing and providing bigger FoVs at the output than
    the FoV at the input.

    Parameters
    ----------
    Uin : Input field (source plane)
    wvl : wavelength of the field
    delta : grid spacing on the input plane (spatial domain)
    z : propagation distance
    zsteps : number of propagation steps
    padsize : size of the padding (if wanted). False by default
    resample_period : if true, resample the field after N propagations.
                        Resample is done by merging 2x2 pixels into a single 
                        pixel (nearest neighbors)

    Returns
    -------
    Uout : Output field (observation plane)
    delta : mesh spacing on the observation plane. Same as 
            input if resample_period = False

    '''
    N = Uin.shape[0] #number of pixels (assume square grid)
    #Padding (if specified)
    if padsize != False:
        #pad the input
        Uin = np.pad(Uin, ((padsize,padsize),(padsize,padsize)), 'constant')
        #Calculate new size in pixels
        N = Uin.shape[0]
    #Define FoV size
    FoV = N*delta
    #Define absorbing window
    window = buildSuperGauss(N, delta, (0,0), (0.35*FoV,0.35*FoV), 4)
    #Define step length
    zSTEP = z/zsteps
    
    #Load phase unwrap + resample methods
    if resample_period != False:
        from skimage.restoration import unwrap_phase
        from PIL import Image
       
    Uout = Uin.copy()
    for idx in range(zsteps):
        #Propagation
        Uout = rs_ang_spec_prop(Uin = Uout, wvl = wvl, delta = delta,
                        z = zSTEP, padsize = False)
        #Apply absorbing window
        Uout *= window
        #Resample (if needed)
        if resample_period != False:
                if (idx > 0) and (idx % resample_period == 0):
                    #Calculate new size
                    M = int(N/2)
                    padsize = int((N-M)/2)
                    #Take amplitude and phase
                    amp = np.abs(Uout)
                    phase = np.angle(Uout)
                    #Resample amplitude
                    amp = Image.fromarray(amp).resize((M,M),
                                                      resample=Image.NEAREST)
                    amp = np.array(amp)
                    #Unwrap phase, then resample it
                    phase = unwrap_phase(phase)
                    phase = Image.fromarray(phase).resize((M,M),
                                                    resample=Image.NEAREST)
                    phase = np.array(phase)
                    #Build field again, pad it to have same size as before
                    Uout = amp*np.exp(1j*phase)
                    Uout = np.pad(Uout, 
                                  ((padsize,padsize),(padsize,padsize)),
                                  'constant')
                    #Update delta
                    delta *= 2
                    #Update FoV size
                    FoV = N*delta
                    #Update absorbing window
                    window = buildSuperGauss(N, delta, (0,0),
                                             (0.35*FoV,0.35*FoV), 4)
    return Uout, delta

def lens_in_front_ft(Uin, wvl, delta, focal, distance, padsize = False):
    '''
    lens_in_front_ft performs propagation of an optical field from a plane 
    in front of a lens, to the focal plane after the lens

    Parameters
    ----------
    Uin : Input field
    wvl : wavelength of the field
    delta : grid spacing input plane
    focal : focal length of the lens
    distance : distance between the objet plane and the plane the lens is placed
    padsize : padding size. The default is False.

    Returns
    -------
    Uout : Field at the focal plane after the lens
    u :  x-grid on the output plane
    v :  y-grid on the output plane

    '''
    N = Uin.shape[0] #number of pixels (assume square grid)
    #Padding (if specified)
    if padsize != False:
        #pad the input
        Uin = np.pad(Uin, ((padsize,padsize),(padsize,padsize)), 'constant')
        #Calculate new size in pixels
        N = Uin.shape[0]
    k = 2*np.pi/wvl #wavenumber
    f_x = np.arange(-N/2, N/2, step = 1) / (N*delta) #frequency axis
    #Generate mesh
    x2, y2 = np.meshgrid(f_x*wvl*focal,-f_x*wvl*focal)
    Uout = 1/(1j*wvl*focal)*np.exp(1j*k/(2*focal)*(1-distance/focal)
                                          *(x2**2+y2**2))*ft2(Uin,delta)
    return Uout, x2, y2

#%% Plotting functions + display/saving. 
'''
This works nice with Spyder to visualize stuff. Not sure at all what will
happen in other IDEs, but the code should be easily patched for those I guess.
'''
def show_img(img, fig_size = False, colormap = 'viridis'):
    '''
    show_img plots a single matrix as an image

    Parameters
    ----------
    img : matrix to plot
    fig_size: size of the figure (inches)
    colormap: colormap of the plot
    Returns
    -------
    ax : ax object (so you have access to it in the workspace)

    '''
    if fig_size == False:
        fig_size = (5,5)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size)
    im1 = ax.imshow(img, interpolation = "nearest", cmap = colormap)    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im1, cax = cax, ax = ax)
    ax.set_aspect(1)
    return ax

def show_2img(img1, img2, fig_size = False, colormap = 'viridis'):
    '''
    show_2img plots two images, side by side

    Parameters
    ----------
    img1 : image #1
    img2 : image #2
    colormap : colormap of the plots
    fig_size : size of the plot window (inches)
    '''
    if fig_size == False:
        fig_size = (10,4)
    fig,(ax1,ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = fig_size)
    im1 = ax1.imshow(img1,interpolation="nearest",cmap = colormap)
    ax1.set_aspect(1)
    ax1_divider = make_axes_locatable(ax1)
    cax1 = ax1_divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im1, cax = cax1)
    im2 = ax2.imshow(img2,interpolation="nearest",cmap = colormap)
    ax2.set_aspect(1)
    ax2_divider = make_axes_locatable(ax2)
    cax2 = ax2_divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im2, cax = cax2)
    pass

def show_Nimg(hypercube, fig_size = False, colormap = 'viridis'):
    '''
    show_Nimg generates a grid plot from a set of 2D images.

    Parameters
    ----------
    hypercube : Set of 2D images. Third axis should be the image number
    fig_size : size of the figure 
    colormap : colormap of the plots

    '''
    if fig_size == False:
        fig_size = (8,8)
    Nimg = hypercube.shape[2]
    nrows = int(np.ceil(np.sqrt(Nimg)))
    fig, ax = plt.subplots(nrows = nrows, ncols = nrows, figsize = fig_size)
    counter = 0
    for rowidx in range(0,nrows):
        for colidx in range(0,nrows):
            if counter < Nimg:
                im = ax[rowidx,colidx].imshow(hypercube[:,:,counter],
                                              cmap = colormap)
                ax[rowidx,colidx].set_aspect(1)
                divider = make_axes_locatable(ax[rowidx,colidx])
                cax = divider.append_axes('right', size='5%', pad = 0.1)
                fig.colorbar(im, cax = cax, ax = ax[rowidx,colidx])
                counter += 1
    plt.tight_layout()
    plt.show()
    pass

def show_vid(hypercube, rate, fig_size = False, colormap='viridis',
             cbarfix = False, loop = False):
    '''
    show_vid creates an animation showing the frames of a video.
    The input is a 3D array, where the third dimension corresponds to time

    Parameters
    ----------
    hypercube : input array, third axis is the frames
    rate : frame rate (in ms)
    fig_size : size of the plot
    colormap : colormap of the plots
    cbarfix : option to have the same colorbar range for all frames (True)
                or not (False)
    Returns
    -------
    anim : animation object (so you have access to it in the workspace, useful
                             for exporting as .mpg or .gif or whatever)


    '''
    import matplotlib.animation as animation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if fig_size == False:
        fig_size = (6,6)
    
    fig , ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size)
    
    if cbarfix == True:
        cmin = np.min(hypercube)
        cmax = np.max(hypercube)
        cbarlimits = np.linspace(cmin,cmax,10,endpoint=True)

    def plot_img(i):
        plt.clf()
        plt.suptitle('Frame #' + str(i))        
        if cbarfix == True:
            im1 = plt.imshow(hypercube[:,:,i],vmin = cmin, vmax = cmax,
                             cmap = colormap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.6)
            ax.set_aspect(1)
            plt.colorbar(im1,cax = cax, ax = ax, ticks = cbarlimits)
        else:
            im1 = plt.imshow(hypercube[:,:,i], cmap = colormap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.6)
            ax.set_aspect(1)
            plt.colorbar(im1,cax = cax, ax = ax)
        plt.show()
    anim = animation.FuncAnimation(fig, plot_img, frames = hypercube.shape[2],
                                   interval = rate, repeat = loop)
    return anim

def plot_scatter2d(data, fig_size = False):
    '''
    plot_scatter plots a 2D scatter plot

    Parameters
    ----------
    data : matrix to plot (row_number is number of points, colum is X and Y)
    fig_size: size of the figure (inches)
    Returns
    -------
    ax : ax object (so you have access to it in the workspace)

    '''
    if fig_size == False:
        fig_size = (5,5)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size)
    ax.scatter(data[:,0],data[:,1])    
    ax.set_aspect(1)
    plt.show()
    return ax

def plot_lineplot(data, fig_size = False,  xlabel = 'x', ylabel = 'y', 
                  title = False, **plot_args):
    '''
    plot_lineplot plots a 2D line plot

    Parameters
    ----------
    data : matrix to plot (row_number is number of points, colum is X and Y)
    fig_size: size of the figure (inches)
    xlabel: name for X axis
    ylabel: name for Y axis
    title: plot title
    **plot_args: kwargs for the plot options (color, markers, whatever u want)
    Returns
    -------
    fig: fig object (so you have access to it in the workspace)
    ax : ax object (so you have access to it in the workspace)
    '''
    if fig_size == False:
        fig_size = (5,5)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size)
    ax.plot(data[:,0], data[:,1], **plot_args)    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.show()
    return fig, ax

def plot_stemplot(data, fig_size = False,  xlabel = 'x', ylabel = 'y', 
                  title = False, linefmt = 'teal', markerfmt = 'D', 
                  markersize = 3):
    '''
    plot_stemplot plots a 2D stem plot (each point has a line going vertically
                                        to the horizontal axis)

    Parameters
    ----------
    data : matrix to plot (row_number is number of points, colums are X and Y
                           values)
    fig_size: size of the figure (inches)
    xlabel: name for X axis
    ylabel: name for Y axis
    title: plot title
    linefmt: color of the stem lines
    markerfmt: marker to use for the data points
    markersize: marker size
    Returns
    -------
    fig: fig object (so you have access to it in the workspace)
    ax : ax object (so you have access to it in the workspace)
    '''
    if fig_size == False:
        fig_size = (5,5)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size)
    markerline, stemlines, baseline = ax.stem(data[:,0], data[:,1], 
                                              linefmt = linefmt, 
                                              markerfmt = markerfmt)
    markerline.set_markerfacecolor('none')
    markerline.set_markersize(markersize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.show()
    return fig, ax

def z2rgb(Z, theme = 'light'):
    '''Takes an array of complex numbers (z) and converts
    it to an array of [r, g, b]. Before it converts z to hsv (makes
    more sense / easier to understand). Phase is encoded in
    hue and saturaton/value are given by the absolute value,
    depeding if you want zero amplitude to be white or black.
    https://en.wikipedia.org/wiki/HSL_and_HSV
    Last step is to convert the HSV color into RGB for representation.
    
    Useful for representing complex fields (amplitude and phase) in a single
    image
    '''
    absmax = np.abs(Z).max() #calculate maximum amplitude
    Y = np.zeros(Z.shape + (3,), dtype='float') #preallocate rgb image
    Y[..., 0] = np.angle(Z) / (2 * np.pi) % 1 #calculate hue (phase)
    #choose either 0 amplitude to be dark or bright
    if theme == 'light':
        #map amplitude to saturation
        Y[..., 1] = np.clip(np.abs(Z) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        #map amplitude to value
        Y[..., 2] = np.clip(np.abs(Z) / absmax, 0, 1)
    #convert HSV colors to RGB
    Y = matplotlib.colors.hsv_to_rgb(Y)
    return Y

def show_field(Z, mode = 'light'):
    '''
    show_field plots a complex array (usually a wavefront).
    Uses amplitude as hue/value, and phase is encoded in color

    Parameters
    ----------
    Z : input field
    mode : map minimum amplitude to black or white (light/dark modes)
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5))
    img = z2rgb(Z, theme = mode)
    im1 = ax.imshow(img, cmap = 'hsv', aspect = Z.shape[1]/Z.shape[0])    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad = 0.1)
    cbar = plt.colorbar(im1, cax = cax, ax = ax, ticks = [0,1])
    cbar.ax.set_yticklabels(['0','2$\pi$'])
    pass

def nparray2png(im_array):
    '''
    nparray2png takes a numPy array and converts it to a
    grayscale image object. Useful for saving results to images

    Parameters
    ----------
    im_array : numpy array representing an image

    Returns
    -------
    img : Image object
    '''
    from PIL import Image
    im = (im_array - np.min(im_array)) / (np.max(im_array) - np.min(im_array)) * 255
    im = im.astype('int8')
    img = Image.fromarray(im, mode = 'L')
    return img

def save_images(imgs, namestring = 'img', leadzeros = 2):
    '''
    save_images stores all  the images from a 3D array (px*px*number_of_images 
    shape) as .png files, or a list of 2D arrays (px*px). Second case (list) 
    can be used to save images with different sizes inside a list

    Parameters
    ----------
    imgs : 3D array with the images stored in the third axis (or list of 
                                                                  2D arrays)
    namestring: string for naming the files ('img' by default)
    leadzeros: number of leading zeros (also for naming the files, 2 by default)

    '''
    if type(imgs) == np.ndarray:
        for idx in range(imgs.shape[2]):
            pic = nparray2png(imgs[:,:,idx])
            pic.save(namestring + '_' + 
                 ('{:0' + str(leadzeros) + 'd}').format(idx) + '.png')
        print('Saving done')
    elif type(imgs) == list:
        for idx in range(len(imgs)):
            pic = nparray2png(imgs[idx])
            pic.save(namestring + '_' +
                  ('{:0' + str(leadzeros) + 'd}').format(idx) + '.png')
        print('Saving done')
    else:
        print(
        'Wrong input format. Provide either 3D nparray or a list of 2D arrays')
    return 

def centralROI(img,size):
    '''
    centralROI crops the central part of an image

    Parameters
    ----------
    img : input image
    size : size of the crop (in pixels)

    Returns
    -------
    croppedIMG : the central part of the image, with a size equals to [size]
    '''
    if img.shape[0]< size or img.shape[1] < size:
        print('Size is bigger than the image size')
        return img
    else:
        center_row = int(img.shape[0]/2)
        center_col = int(img.shape[1]/2)
        semiROI = int(size/2)
        croppedIMG = img[center_row - semiROI : center_row + semiROI,
                         center_col - semiROI : center_col + semiROI]
    return croppedIMG

def cropROI(img,size,center_pos):
    '''
    cropROI gets a ROI with a desired size, centered at a fixed position

    Parameters
    ----------
    img : input image
    size : size of the ROI (2 element vector, size in [rows,cols] format)
    center_pos : central position of the ROI

    Returns
    -------
    cropIMG = cropped ROI of the image

    '''
    if img.shape[0]< size[0] or img.shape[1] < size[1]:
        print('Size is bigger than the image size')
        return img
    else:
        center_row = center_pos[0]
        center_col = center_pos[1]
        semiROIrows = int(size[0]/2)
        semiROIcols = int(size[1]/2)
        cropIMG = img[center_row - semiROIrows : center_row + semiROIrows,
                      center_col - semiROIcols : center_col + semiROIcols]
    return cropIMG

def ROIclick(img):
    '''
    ROIclick lets you generate a ROI from an image (numpy array) 
    Opens the image in a window, lets you select with the mouse 
    You need to press any key to end the process

    Parameters
    ----------
    img : Input image (numpy array)

    Returns
    -------
    roi : Cropped image
    roiData : Position and size of the ROI, as a tuple with 4 elements
        roiData[0]: col start of the ROI
        roiData[1]: row start of the ROI
        roiData[2]: col size of the ROI
        roiData[3]: row size of the ROI

    '''
    temp_img = np.copy(img)
    temp_img2 = np.array(nparray2png(temp_img))
    temp_img_color = cv2.applyColorMap(temp_img2, cv2.COLORMAP_VIRIDIS)
    cv2.namedWindow('image', flags = cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.imshow('image', temp_img_color)
    showCrosshair = True
    fromCenter = False
    roiData  = cv2.selectROI("image", temp_img_color, showCrosshair, fromCenter)
    roi = temp_img[roiData[1] : roiData[1] + roiData[3],
                   roiData[0] : roiData[0] + roiData[2]]
    cv2.destroyWindow('image')
    return roi, roiData

def ROIremove(img, fill = 0):
    '''
    ROIremove lets you remove parts of an image (numpy array)
    Opens the image in a window, lets you select a ROI with the mouse, then 
    you press ENTER and it shows you the image without that ROI. It asks if you
    want to keep removing ROIs, or if its over. If that is the case, it returns
    you your image without the ROIs.

    Parameters
    ----------
    img : Input image (numpy array)
    fill : constant to fill the ROI with. Zeros by default

    Returns
    -------
    img_crop : Cropped image

    '''
    keep_cropping = True
    showCrosshair = True
    fromCenter = False
    temp_img = np.copy(img)#create temporal copy of the image to manipulate
    while keep_cropping:
        #convert so OpenCV can display
        temp_img2 = np.array(nparray2png(temp_img))
        #generate a copy with colormap for easier visualization
        temp_img_color = cv2.applyColorMap(temp_img2, cv2.COLORMAP_VIRIDIS)
        #create window
        cv2.namedWindow('image', flags = cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        #plot image with the colormap
        cv2.imshow('image', temp_img_color)
        #let user select the ROI
        roiData  = cv2.selectROI("image", temp_img_color, showCrosshair, fromCenter)
        #set ROI to zero in both the image to display and the image to return
        temp_img[roiData[1] : roiData[1] + roiData[3],
                 roiData[0] : roiData[0] + roiData[2]] = fill
        temp_img2[roiData[1] : roiData[1] + roiData[3],
                 roiData[0] : roiData[0] + roiData[2]] = fill
        #plot image with the ROI set to zero, ask whether to keep going or stop
        temp_img_color = cv2.applyColorMap(temp_img2, cv2.COLORMAP_VIRIDIS)
        cv2.imshow('image', temp_img_color)
        keep_cropping = input("Keep cropping? (Y/N): ").casefold() == 'Y'.casefold()
    cv2.destroyWindow('image')#close window
    return temp_img
    

#%% Useful tools for general optics simulations
''' 
Generate phase profiles, masks, apertures, etc. Useful to represent lenses and 
doing step-by-step optical propagations, defining apertures and studying
diffraction, etc.
'''
def circAp(totalSize, radius):
    """
    circAp generates a circular aperture, given the total size of the mask and
    the radius of the aperture (between 0 and 1). Also provides the coordinates
    of the mask in polar coordinate system
    Parameters
    ----------
    totalSize : int
        size of the mask, in pixels.
    radius : float
        desired size of the radius (between 0 and 1).
    Returns
    -------
    mask : logical 2D array
        mask with the circular aperture
    r : float 2D array
        polar coordinates (radius)
    theta : float 2D array
        polar coordinates (angle)
    """
    x = np.linspace(-1,1,totalSize) #Define x axis
    y = np.linspace(1,-1,totalSize) #Define y axis
    xx, yy = np.meshgrid(x,y)   #Generate mesh
    #Generate radius (polar)
    r = np.abs(xx + 1j*yy)
    #Generate angle (polar)
    theta = np.angle(xx + 1j*yy)
    #Generate mask with circular aperture of given size
    mask = r<radius
    
    return mask,r,theta

def phaseRamp(px, elevation, angle):
    '''
    phaseRamp generates a ramp phase mask (useful to introduce tip/tilt into
                                           a wavefront, for example)

    Parameters
    ----------
    px : size of the image, in pixels
    elevation : Total value of the phase ramp (times 2pi)
    angle : orientation of the ramp, in radians (0 = horizontal, 
                                                 pi/2 = vertical)

    Returns
    -------
    ramp : ramp image

    '''
    #build ramp image
    x = np.linspace(-1,1,px) #axes
    x,y = np.meshgrid(x,-x) #axes
    ramp = x*np.cos(angle) + y*np.sin(angle) #ramp profile
    #build complex mask
    phaseRamp = np.exp(1j*np.pi*ramp*elevation)
    return phaseRamp

def buildGauss(px, sigma, center, phi):
    """
    buildGauss generates a Gaussian profile in 2D. Formula from
    https://en.wikipedia.org/wiki/Gaussian_function

    Parameters
    ----------
    px : image size of the output (in pixels)
    sigma : 2-element vector, sigma_x 
    and sigma_y for the 2D Gaussian
    center : 2-element vector, center position
    of the Gaussian in the image
    phi : Rotation angle for the Gaussian

    Returns
    -------
    gaus : 2D image with the Gaussian

    """
    #Generate mesh
    x = np.linspace(1,px,px)
    X,Y = np.meshgrid(x,x)
    
    #Generate gaussian parameters
    a = np.cos(phi)**2/(2*sigma[0]**2) + np.sin(phi)**2/(2*sigma[1]**2)
    b = -np.sin(2*phi)/(4*sigma[0]**2) + np.sin(2*phi)/(4*sigma[1]**2)
    c = np.sin(phi)**2/(2*sigma[0]**2) + np.cos(phi)**2/(2*sigma[1]**2)
    
    #Generate Gaussian
    gaus = np.exp(-(a*(X-center[0])**2 +
                    2*b*(X-center[0])*(Y-center[1]) +
                    c*(Y-center[1])**2))
    
    return gaus

def buildSuperGauss(px, delta, position, sigma, P):
    """
    buildSuperGauss generates a superGaussian function in 2D. Formula from
    https://en.wikipedia.org/wiki/Gaussian_function, to the power of P

    Parameters
    ----------
    px : image size of the output (in pixels)
    delta : mesh grid spacing (m)
    position : position of the window (m)
    sigma : 2-element vector, sigma_x 
    and sigma_y for the 2D Gaussian (m)
    P : power to build the superGaussian 

    Returns
    -------
    supergaus : 2D image with the superGaussian

    """
    #Generate mesh
    x = np.arange(-px/2,px/2,step=1)*delta #x-axis
    X, Y = np.meshgrid(x,-x) #generate grid
    #Generate superGaussian
    t1 = (X-position[0])**2/(2*sigma[0]**2)
    t2 = (Y-position[1])**2/(2*sigma[1]**2)
    supergaus = np.exp(-(t1+t2)**P)
    
    return supergaus

def filt_fourier(img, filt_func):
    '''
    filt_fourier filters an image in the Fourier domain.
    To do so, it uses [filt_func]. It multiplies that mask to the 
    Fourier transform of the input image [img], thus eliminating some 
    frequency content. Then it goes back to image domain.

    Parameters
    ----------
    img : Input image (to be filtered)
    filt_func : Filtering mask in the Fourier domain
    
    Returns
    -------
    img_filt : Filtered image

    '''
    # Go to Fourier domain
    img_k = fftshift(fft2(fftshift(img)))
    # Apply filter
    img_k_filt = img_k*filt_func
    # Go back to image domain
    img_filt = np.abs(ifftshift(ifft2(ifftshift(img_k_filt))))
    return img_filt

def gen_quadphase_paraxial(px, size, focal, wavelength):
    '''
    gen_quadphase_paraxial generates a quadratic phase profile that represents the 
    phase introduced by a thin lens with a given focal distance.
    Based on the thin lens maker formula. Should be valid for paraxial
    approximation 

    Parameters
    ----------
    px : size of the lens (in pixels)
    size : size of the lens (in m)
    focal : focal distance (m)
    wavelength : wavelength of the beam going through the lens

    Returns
    -------
    phase_profile

    '''
    delta = size/px
    #Generate mesh
    x = np.arange(-0.5*px,0.5*px,1)*delta
    X,Y = np.meshgrid(x,-x)
    #Generate phase
    phase_profile = np.exp(-1j*2*np.pi/wavelength*1/(2*focal)*(X**2+Y**2))
    
    return phase_profile

def gen_quadphase(px, size, f, wavelength):
    '''
    gen_quadphase generates a quadratic phase profile that cancels out the
    phase of a spherical wavefront at a distance = z.
    It is a different way of thinking about a lens (that should convert a
    spherical wavefront into a plane wave). This should be valid for 
    high angle (non-paraxial) simulations, I think.

    Parameters
    ----------
    px : size of the lens (in pixels)
    size : size of the lens (in m)
    f : focal distance (m) / propagation distance along optical axis
    wavelength : wavelength of the beam going through the lens

    Returns
    -------
    phase_profile

    '''
    delta = size/px
    #Generate mesh
    x = np.arange(-0.5*px, 0.5*px, 1)*delta
    X,Y = np.meshgrid(x,-x)
    r = np.sqrt((X)**2 + (Y)**2 + (f)**2)
    #Generate phase
    phase_profile = np.exp(-1j*2*np.pi/wavelength*1*r)
    
    return phase_profile

def noisify(signal,end_snr):
    '''
    Add white gaussian noise to a signal so it ends with end_snr 
    signal-to-noise ratio.
    
    Parameters
    ----------
    signal : input signal.
    end_snr : desired SNR.
    Returns
    -------
    noisy_signal : signal with added noise.
    noise : noise added to the original signal.
    '''
    # calculate signal average (~power)
    signal_avg = np.mean(signal)
    # convert to dB
    signal_avg_db = 10 * np.log10(signal_avg)
    # calculate noise power in dB, using objective SNR
    #SNR = P_signal(dB) - P_noise(dB)
    noise_avg_db = signal_avg_db - end_snr
    # convert to noise average (~power)
    noise_avg = 10**(noise_avg_db/10)
    # Build noise with desired power
    noise = np.random.normal(0,np.sqrt(noise_avg),signal.size)
    # Build additive noise (shift to positive-only values)
    noise = noise + np.abs(np.min(noise))
    # Add noise to signal
    noisy_signal = signal + noise
    
    return noisy_signal, noise
###############################################################################
############################# Comparing images ################################
###############################################################################
def corr2(A,B):
    '''
    correlation coefficient between two images (2d numpy arrays)

    Parameters
    ----------
    A : image 1
    B : image 2

    Returns
    -------
    corr2
    '''
    num = np.sum((A - np.mean(A)) * (B - np.mean(B)))
    deno = np.sqrt(np.sum((A - np.mean(A)) ** 2) * np.sum((B - np.mean(B)) ** 2))
    return num/deno

def xcorr2d(im1, im2, norm = False, mode = 'same'):
    '''
    Takes into account that the two images might have 
    different sizes (and padding the smaller dimensions to solve it).
    
    mode controls wether the size of the xcorr is the full one (2 times the 
    biggest size of the two images) or the central part (size equals the 
    size of the biggest image)
    
    Even though the code works with odd sizes, ft2 gives problems when 
    working with those. Even sizes should be prefered!

    Parameters
    ----------
    im1 : input image
    im2 : input image
    norm : whether or not to normalize the correlation
    mode : 'full' or 'same'

    Returns
    -------
    output image (cross-correlation between im1 and im2)

    '''
    rows1, cols1 = im1.shape
    rows2, cols2 = im2.shape
    
    if rows1 != rows2 or cols1 != cols2:
        rowsmax = max(rows1,rows2)
        colsmax = max(cols1,cols2)
        #fix image 1
        rows1diff = rowsmax - rows1
        cols1diff = colsmax - cols1
        im1 = np.pad(im1,((int(np.floor(rows1diff/2)),
                     int(np.ceil(rows1diff/2))),
                    (int(np.floor(cols1diff/2)),int(np.ceil(cols1diff/2)))),
                   'constant')
        #fix image 2
        rows2diff = rowsmax - rows2
        cols2diff = colsmax - cols2
        im2 = np.pad(im2,((int(np.floor(rows2diff/2)),
                     int(np.ceil(rows2diff/2))),
                    (int(np.floor(cols2diff/2)),int(np.ceil(cols2diff/2)))),
                   'constant')
    
    if mode == 'full':
        rows1, cols1 = im1.shape
        rows2, cols2 = im2.shape
        im1 = np.pad(im1,((int(rows2),int(rows2)),(int(cols2),int(cols2))),
                     mode = 'constant')
        im2 = np.pad(im2,((int(rows1),int(rows1)),(int(cols1),int(cols1))),
                     mode = 'constant')
        
    delta = 1
    delta_f = 1/(im1.shape[0]*delta)
    xcorr2d = np.real(ift2(np.conj(ft2(im1, delta)) * ft2(im2, delta), delta_f))
    if norm:
        xcorr2d /= (np.linalg.norm(im1) * np.linalg.norm(im2))
    return xcorr2d

def snr_img(img):
    '''
    snr_img returns the SNR of an image, calculated as the ratio between
    the mean value of the image and its standard deviation

    Parameters
    ----------
    img : input image

    Returns
    -------
    snr : signal to noise ratio

    '''
    mean = np.mean(np.matrix.flatten(img))
    std = np.std(np.matrix.flatten(img))
    snr = mean/std
    return snr

#%% Light field / wavefront sensing routines. 
'''
Crop raw images into elemental images (for individual manipulation). Calculate 
centroid positions from MLA images, calculate wavefronts from centroid
positions, decompose wavefronts in Zernike basis, etc.
'''

def cropFOV(input_image, num_crops):
    '''
    Crops an image into subimages (squares)

    Parameters
    ----------
    input_image : image to crop
    num_crops : lateral number of crops (vector with number of rows and cols)

    Returns
    -------
    crops : Images (cropped)
    '''
    #Define size of the crop
    subimg_size = int(input_image.shape[0]/num_crops[0])
    #Initialize crops
    crops = np.zeros((subimg_size,subimg_size,num_crops[0],num_crops[1]))
    #Crop the image into smaller ones
    for idxrows in range(0,num_crops[0]):
        for idxcols in range(0,num_crops[1]):
            crops[:,:,idxrows,idxcols] = input_image[idxrows*subimg_size : 
                        (idxrows+1)*subimg_size, idxcols*subimg_size :
                            (idxcols+1)*subimg_size]
    return crops

def uncropFOV(input_images):
    '''
    Merges cropped images into big image

    Parameters
    ----------
    input_images : Cropped images

    Returns
    -------
    uncrop : Image composed from all the cropped images
    '''
    #Get number of cropped images
    num_crops = np.array((input_images.shape[2],input_images.shape[3]))
    #Get size of each individual image
    size_crops = np.array((input_images.shape[0],input_images.shape[1]))
    #stitch together all the images
    uncrop = np.zeros((input_images.shape[0]*num_crops[0],
                       input_images.shape[1]*num_crops[1]))
    for idxrow in range(num_crops[0]):
        for idxcol in range(num_crops[1]):
            uncrop[idxrow*size_crops[0]:(idxrow+1)*size_crops[0],
                   idxcol*size_crops[1]:(idxcol+1)*size_crops[1]] = input_images[:,:,idxrow,idxcol]
    return uncrop

def raw2centroid(raw_img, num_lenslets):
    '''
    raw2centroid takes a raw MLA image and calculates the position of the 
    centroid for every microimage.

    Parameters
    ----------
    raw_img : raw MLA image
    num_lenslets : MLA shape: (rows,cols) microlenses

    Returns
    -------
    centroid_img : image with all the centroids
    centroids : centroid coordinates inside every microimage
    centroid_energy : energy of every microimage (might be useful at some point)

    '''
    #preallocate 
    centroids = np.zeros((num_lenslets[0], num_lenslets[1], 2)).astype('float')
    crops = cropFOV(raw_img,num_lenslets) #crop input image in subimages
    subimgs = np.zeros((crops.shape[0], crops.shape[1], crops.shape[2],
                        crops.shape[3])) #preallocate
    centroid_energy = np.zeros((num_lenslets[0],num_lenslets[1],1))#preallocate
    for idxrow in range(0,num_lenslets[0]):
        for idxcol in range(0,num_lenslets[1]):
            #calculate centroid position
            if np.sum(crops[:,:,idxrow,idxcol]) != 0:
                centroids[idxrow,idxcol,:] = np.array(sc.ndimage.measurements.center_of_mass(crops[:,:,idxrow,idxcol])).astype('int')
            else:
                centroids[idxrow,idxcol,:] = np.array((int(crops[:,:,0,0].shape[0]/2),
                                                       int(crops[:,:,0,0].shape[1]/2)))
            #calculate energy of subimage
            centroid_energy[idxrow,idxcol] = np.sum(crops[:,:,idxrow,idxcol])
            #generate image with one pixel set to 1 at centroid position
            subimgs[:,:,idxrow,idxcol] = np.zeros((crops.shape[0], crops.shape[1]))
            subimgs[int(centroids[idxrow,idxcol,0]),
                    int(centroids[idxrow,idxcol,1]), idxrow,idxcol] = 1
            pass
        pass
    centroid_img = uncropFOV(subimgs)
    return centroid_img, centroids, centroid_energy

def corrApert(full_img, num_crops, subimg_idx, sigma_gauss):
    '''
    corrApert takes one image and crops into patches. Then takes one patch and
    correlates with all the patches. Finds the maximum of the correlation 
    and returns an image with its position.

    Parameters
    ----------
    full_img : full image to obtain sub correlations
    num_crops : number of crops (regions) inside the image
    subimg_idx : index of the image to correlate with
    sigma_gauss : size of the gaussian filter (to remove the envelope in the 
                                               xcorr images)
    Returns
    -------
    xcor : image with all the correlations
    xcorpeak : same as xcor, but only the positions of the peaks (without the 
                envelope)

    '''
    #Crop input image
    cropped_img = cropFOV(full_img,num_crops)
    #Take reference image
    test_img = cropped_img[subimg_idx]
    xcor = {} #initialize cross-correlation images
    xcorpeak = {} #initialize positions of the cross-correlation peaks
    #Generate gaussian. Needed to filter cross-correlation and finding the peaks
    gaussfilt = 1 - buildGauss(px=test_img.shape[0],sigma=(sigma_gauss,sigma_gauss),
                               center=(int(test_img.shape[0]/2),int(test_img.shape[1]/2)),
                               phi=0)
    #Do the cross-correlation betwen reference image and all the images
    for idx in range(0,num_crops**2):
        #correlation
        xcor[idx] = sc.signal.correlate(test_img,cropped_img[idx],mode='same',method='fft')
        xcorpeak[idx] = filt_fourier(xcor[idx],gaussfilt) #filer low-frequency
        #Find the peak
        maxpos = np.unravel_index(np.argmax(xcorpeak[idx]),xcorpeak[idx].shape)
        #Generate black image, set the peak position to 1
        xcorpeak[idx] = np.zeros(xcorpeak[idx].shape)
        xcorpeak[idx][maxpos] = 1
        pass
    #Merge images into one
    xcor = uncropFOV(xcor)
    xcorpeak = uncropFOV(xcorpeak)
    return xcor,xcorpeak

def subap2lowres(subap_img, pxnum):
    '''
    subap2lowres takes a subaperture image and adds all the values of the pixels
    inside each subimage, to get a low resolution image of the scene. This would
    be the image you get if you take the MLA and put it onto the image plane, so
    each microlens samples a region of the object, and you add all the pixels
    corresponding to each microlens in you sensor

    Parameters
    ----------
    subap_img : Subaperture image
    pxnum : pixel number (lateral) for each microlens image

    Returns
    -------
    lowres_img : locally integrated subaperture image

    '''
    lowres_img = cropFOV(subap_img,pxnum)
    for idx in range(0,len(lowres_img)):
        lowres_img[idx] = np.sum(lowres_img[idx])
        pass
    lowres_img = np.array(list(lowres_img.items()))[:,1]
    lowres_img = np.reshape(lowres_img,(pxnum,pxnum))
    return lowres_img

def elementalCorrelations(elemental_img, num_img, sigma_gauss):
    '''
    elementalCorrelations takes an image with contains elemental images, 
    i.e. images from the same scene but seen from different perspectives, and
    calculates the cross-correlation peak position (from the center of the image)
    between neighbors. Right now this is done by searching for the maximum in
    the cross-correlation images (we assume the maximum value is the peak, and
    find its position). This seems to work for now (as images tend to be 
    very simmilar, because they are close neighbors). If this stops working,
    we can high-pass filter the cross correlation images and then take the max,
    as that should be the peak.

    Parameters
    ----------
    elemental_img : input elemental image
    num_img : number of elemental images inside the input
    sigma_gauss : width of the gaussian for filtering the cross-correlation
                    images, in order to find the peak position 
                    (we high-pass the cross-correlations to remove the envelope)
    Returns
    -------
    gradX : horizontal positions of the cross-correlation peaks
    gradY : vertical positions of the cross-correlation peaks

    '''
    #Crop the elemental image into subimages
    crops = cropFOV(elemental_img, num_img)
    #Convert to array form (instead of list) for easier manipulation
    sub_img = np.zeros((num_img,num_img,crops[0].shape[0],crops[0].shape[1]))
    k = 0
    for ridx in range(0,num_img):
        for cidx in range(0,num_img):
            sub_img[ridx,cidx,:,:] = crops[k]
            k += 1
    
    #Perform cross-correlations, take the maximum as displacement for gradient
    gradX = np.zeros((num_img,num_img)) #preallocating
    gradY = np.zeros((num_img,num_img)) #preallocating
    
    #Generate gaussian. Needed to filter cross-correlation and finding the peaks
    gaussfilt = 1 - buildGauss(px=sub_img.shape[2],sigma=(sigma_gauss,sigma_gauss),
                               center=(int(sub_img.shape[2]/2),int(sub_img.shape[2]/2)),
                               phi=0)
  
    
    #X-axis partial derivatives
    for ridx in range(0,num_img):
        for cidx in range(1,num_img):
            #correlate image with the one on its left
            temp = sc.signal.correlate(sub_img[ridx,cidx,:,:],
                                       sub_img[ridx,cidx-1,:,:],
                                       mode='same',method='fft')
            #take the correlation peak lateral position (with respect from the 
            #center of the image) as the value of the gradient. Filter low
            #frequency content, then find the maximum (should correspond to the
            #peak)
            temp = filt_fourier(temp,gaussfilt)
            gradX[ridx,cidx] = np.where(temp==np.max(temp))[1] - temp.shape[1]/2

    #Y-axis derivatives
    for ridx in range(1,num_img):
        for cidx in range(0,num_img):
            #correlate image with the one on top
            temp = sc.signal.correlate(sub_img[ridx,cidx,:,:],
                                       sub_img[ridx-1,cidx,:,:],mode='same',
                                       method='fft')
            #take the correlation peak lateral position (with respect from the 
            #center of the image) as the value of the gradient. Filter low
            #frequency content, then find the maximum (should correspond to the
            #peak)
            temp = filt_fourier(temp,gaussfilt)
            gradX[ridx,cidx] = np.where(temp==np.max(temp))[1] - temp.shape[1]/2
    
    return gradX,gradY

###############################################################################
######################## Zernike related stuff ################################
###############################################################################
'''
Generate Zernike polynomials, express images in Zernike basis, calculate
fields from gradients using Zernike, etc.
For Zernike generation, I used some functions from the AOtools library:
https://github.com/AOtools 
'''
def iseven(number):
    '''
    Should be pretty easy to see what this does

    Parameters
    ----------
    number : input number
    Returns
    -------
    True/False if the number is even/odd

    '''    
    return number % 2 == 0

def isodd(number):
    '''
    Should be pretty easy to see what this does

    Parameters
    ----------
    number : input number
    Returns
    -------
    True/False if the number is odd/even

    '''    
    return number % 2 != 0

def wave2zernike(wavefront, terms):
    '''
    This function generates the Zernike polynomials, and expands a wavefront
    in the basis of Zernike Polynomials by solving the matrix system

    Parameters
    ----------
    wavefront : field to express as Zernike modes linear comb
    terms : number of modes to use

    Returns
    -------
    zernikeExpansion: Zernike coefficients

    '''
    # Creation of the wavefront vector, W. This is simply the wavefront
    # expressed in a column vector instead of a square matrix. 
    [p, q] = wavefront.shape
    W = np.resize(wavefront, (p**2,1))
    
    # Creation of the [M] matrix. First we generate the Zernike polynomials and
    # then we put them in matrix form

    #Generate zernike polynomials
    zernikes = zernikeArray(terms,p)
    # Generation of the F matrix. Rearranging each Zernike function into a
    # column vector and putting them all together onto a matrix
    F = np.zeros((p**2,terms)) #preallocating
    for idx in range(0,terms):
        F[:,idx] = np.reshape(zernikes[idx,:,:],(p**2,))

    #Pseudoinverse generation
    Finv = np.linalg.pinv(F)
    #Expansion coefficients generation
    zernikeExpansion = Finv @ W
    
    return zernikeExpansion

def zernIndex(j):
    """
    Find the [n,m] list giving the radial order n and azimuthal order
    of the Zernike polynomial of Noll index j.

    Parameters:
        j (int): The Noll index for Zernike polynomials

    Returns:
        list: n, m values
    """
    n = int((-1.+np.sqrt(8*(j-1)+1))/2.)
    p = (j-(n*(n+1))/2.)
    k = n%2
    m = int((p+k)/2.)*2 - k

    if m!=0:
        if j%2==0:
            s=1
        else:
            s=-1
        m *= s

    return [n, m]

def zernikeRadialFunc(n, m, r):
    """
    Fucntion to calculate the Zernike radial function

    Parameters:
        n (int): Zernike radial order
        m (int): Zernike azimuthal order
        r (ndarray): 2-d array of radii from the centre the array

    Returns:
        ndarray: The Zernike radial function
    """

    R = np.zeros(r.shape)
    for i in range(0, int((n - m) / 2) + 1):

        R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                         np.math.factorial(n - i)) /
                         (np.math.factorial(i) *
                          np.math.factorial(0.5 * (n + m) - i) *
                          np.math.factorial(0.5 * (n - m) - i)),
                         dtype = 'float')
    return R

def zernike_nm(n, m, N, pupil = True):
    """
     Creates the Zernike polynomial with radial index, n, 
     and azimuthal index, m.

     Args:
        n (int): The radial order of the zernike mode
        m (int): The azimuthal order of the zernike mode
        N (int): The diameter of the zernike more in pixels
        pupil (bool): Either to pupil or not the mode
     Returns:
        ndarray: The Zernike mode
     """
    coords = (np.arange(N) - N / 2. + 0.5) / (N / 2.)
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    if m==0:
        Z = np.sqrt(n+1)*zernikeRadialFunc(n, 0, R)
    else:
        if m > 0: # j is even
            Z = np.sqrt(2*(n+1)) * zernikeRadialFunc(n, m, R) * np.cos(m*theta)
        else:   #i is odd
            m = abs(m)
            Z = np.sqrt(2*(n+1)) * zernikeRadialFunc(n, m, R) * np.sin(m*theta)
    if pupil != False:
        # clip
        Z = Z*np.less_equal(R, 1.0)
        mask, _, _ = circAp(N,1)
        Z *= mask
    
    return Z

def zernike_noll(j, N, pupil = True):
    """
     Creates the Zernike polynomial with mode index j,
     where j = 1 corresponds to piston.

     Args:
        j (int): The noll j number of the zernike mode
        N (int): The diameter of the zernike more in pixels
        pupil (bool): Either to pupil or not the mode
     Returns:
        ndarray: The Zernike mode
     """
    n, m = zernIndex(j)
    return zernike_nm(n, m, N, pupil)

def zernikeArray(J, N, norm = "noll", pupil = True):
    """
    Creates an array of Zernike Polynomials

    Parameters:
        maxJ (int or list): Max Zernike polynomial to create, or list 
                            of zernikes J indices to create
        N (int): size of created arrays
        norm (string, optional): The normalisation of Zernike modes. 
                                Can be "noll", "p2v" (peak to valley),
                                or "rms". default is "noll".
        pupil (bool): Either to pupil or not the mode
    Returns:
        ndarray: array of Zernike Polynomials
    """
    # If list, make those Zernikes
    try:
        nJ = len(J)
        Zs = np.empty((nJ, N, N))
        for i in range(nJ):
            Zs[i] = zernike_noll(J[i], N, pupil)
    # Else, cast to int and create up to that number
    except TypeError:
        maxJ = int(np.round(J))
        N = int(np.round(N))
        Zs = np.empty((maxJ, N, N))
        for j in range(1, maxJ+1):
            Zs[j-1] = zernike_noll(j, N, pupil)
    if norm=="p2v":
        for z in range(len(Zs)):
            Zs[z] /= (Zs[z].max()-Zs[z].min())
    elif norm=="rms":
        mask,_,_ = circAp(N,1)
        for z in range(len(Zs)):
            # Norm by RMS. Remember only to include circle elements in mean
            Zs[z] /= np.sqrt(np.sum(Zs[z]**2)/np.sum(mask))
    
    return Zs

def grad2zernike(gradX, gradY, num_coefs):
    '''
    grad2zernike generates the zernike decomposition from a set of partial
    derivatives of a wavefront.

    Parameters
    ----------
    gradX : gradient in X direction
    gradY : gradient in Y direction
    num_coefs : Number of zernike polynomials to calculate
    Returns
    -------
    zernCoefs : zernike expasion of the wavefront

    '''
    #Calculate Zernike modes (without circular pupil)
    zernikes = zernikeArray(num_coefs,gradX.shape[0],pupil=False)
    #Preallocate Zernike partial derivatives
    zernX = np.zeros((num_coefs,gradX.shape[0],gradX.shape[1]))
    zernY = np.zeros((num_coefs,gradY.shape[0],gradY.shape[1]))
    #Calculate derivatives, put the pupil
    mask,_,_ = circAp(zernX.shape[1],1)
    for idx in range(0,num_coefs):
        zernX[idx,:,0:-1] = (zernikes[idx,:,1:]-zernikes[idx,:,0:-1:])
        zernX[idx,:,:] *= mask
        zernY[idx,0:-1,:] = (zernikes[idx,1:,:]-zernikes[idx,0:-1:,:])
        zernY[idx,:,:] *= mask
        pass
    #Put partial derivatives in matrix form
    Zx = np.zeros((gradX.size,num_coefs))
    Zy = np.zeros((gradY.size,num_coefs))
    for idx in range(0,num_coefs):
        Zx[:,idx] = np.reshape(zernX[idx,:,:],(gradX.size,))
        Zy[:,idx] = np.reshape(zernY[idx,:,:],(gradY.size,))
        pass
    #Create full derivative matrix
    Z = np.append(Zx,Zy,axis=0)
    
    #Create observation vector (from gradients)
    y1 = np.reshape(gradX,(gradX.size,1))
    y2 = np.reshape(gradY,(gradY.size,1))
    y = np.append(y1,y2,axis=0)
    
    #Calculate pseudoinverse of Z
    Zinv = np.linalg.pinv(Z)
    #Calculate Zernike coefs
    zernCoefs = Zinv @ y
        
    return zernCoefs

def zernike2wave(modes, pxNum):
    '''
    zernike2wave generates a surface, given a set of zernike coefficients

    Parameters
    ----------
    modes : vector of weights of the zernike expansion
    pxNum : desired size of the surface

    Returns
    -------
    wavefront : surface

    '''
    #find relevant zernike modes (not 0)
    zernIndexes = np.where(np.abs(modes)>0)[0]
    #Generate zernike polynomials for those modes
    zernikes = zernikeArray(zernIndexes+1,pxNum)
    #preallocate wavefront
    wavefront = np.zeros((pxNum,pxNum))
    #Generate wavefront as linear combination of modes
    for idx in range(0,zernIndexes.size):
        wavefront += zernikes[idx,:,:] * modes[zernIndexes[idx]]
    return wavefront

def gradient(img, delta):
    '''
    gradient calculates the gradient of an image in both X and Y directions

    Parameters
    ----------
    img : image to calculate the gradient
    delta : mesh spacing

    Returns
    -------
    grad : tensor with the gradient in X and Y directions

    '''
    grad = np.zeros((2, img.shape[0], img.shape[1]))
    grad[0,:,0:-1] = np.diff(img, axis = 1) / delta
    grad[1,0:-1,:] = np.diff(img, axis = 0) / delta
    return grad

#%% Brain activity simulations (neuron spikes)

def act_pulse(ratio_ud, dec_rate, fSize):
    '''
    Generate an activation pulse for a neuron. Linear (ramp) rise, 
    then reaches maximum value (1), then exponential fall (to ~0)
    Parameters
    ----------
    ratio_ud : ratio between the increasing part of the pulse and 
    the decreasing part. This makes the activation be faster or 
    slower (increases slope of the uprise)
    dec_rate : decay rate for the exponential decay
    fSize : final size of the pulse (rescaling effect)
    Returns
    -------
    pulse : neuron activation pulse
    '''
    from skimage.transform import resize
    
    #size of the building block for the window. Has to be big enough (resolution).
    ker_size = 1024
    #Define size of the rising part of the pulse
    up_size = int(ker_size*ratio_ud)
    #Build rising part of the pulse
    up = np.linspace(0,1,up_size)
    #Build falling part of the pulse (exp. decay)
    down = np.exp(-dec_rate*np.linspace(0,1,ker_size-up_size))
    #Concatenate to build full pulse
    pulse = np.concatenate((up[:-1],down))
    #Reshape to desired length
    pulse = resize(pulse,(fSize,),preserve_range=True,mode='edge')
    
    return pulse

def temp_trace(pulse_num, length, max_power):
    '''
    Generates a temporal trace (multiple activation pulses) of a neuron
    Parameters
    ----------
    pulse_num : number of pulses inside the trace
    length : duration of the temporal trace
    max_power : maximum value for the pulses
    Returns
    -------
    trace : temporal trace of the neuron
    '''
    #Define pulse width
    pwidth = int(length/(pulse_num*3))
    trace = np.zeros((length,)) #preallocate trace
    #preallocate temporal mesh
    indexes = np.arange(0, length, 1, dtype = 'int')
    #preallocate vector of starting positions for the pulses
    start_pos = np.zeros((pulse_num,), dtype = 'int')
    #Define forbidden starting positions (not to close to the end
    #of the trace)
    idx_remove = np.arange(length-pwidth,length,1,dtype='int')
    #Generate pulse positions
    for cidx in range(pulse_num):
        #pick indexes that are not forbidden
        indexes_new = np.setxor1d(indexes,idx_remove)
        #pick one random index in the whitelist for a pulse to start at
        start_pos[cidx] = indexes_new[np.random.randint(0,np.size(indexes_new))]
        #Update list of forbidden times
        '''
        No pulses can be put if they would overlap with an 
        old pulse or too close to the end
        '''
        start_idx = start_pos[cidx]-pwidth
        if start_idx<0:
            start_idx=0
        #Update forbidden list of times
        idx_remove = np.concatenate((idx_remove,
                        np.arange(start_idx, start_pos[cidx] + pwidth, 1, 
                                  dtype = 'int')))
    #Sort starting positions for the pulses
    start_pos = np.sort(start_pos)[::-1]
    #Generate pulses at given times
    for idx in range(pulse_num):
        #define random power for the pulse. Make sure its not too small
        power = 1e-4*float(np.random.randint(0.1*1e4, 1e4, 1))*max_power
        #generate pulse in each segment of the trace
        growRatio = np.random.randint(1,5)/100
        decay_rate = np.random.randint(int(pwidth/10), int(pwidth/3))
        trace[start_pos[idx]:start_pos[idx]+pwidth] = act_pulse(growRatio,
                                                        decay_rate,pwidth)*power

    return trace

#%% Miscellaneous 
def cropSpeckle(speckle, filtersize, filterLevel, resizing = False):
    '''
    cropSpeckle takes a complex speckle field, finds its central part by 
    looking at its envelope, and returns that central part. The size
    around that central part is defined as a ratio between the maximum
    of the envelope a predefined number (input by user) 
    Size of the image can be the crop size, or a resizing to the next 
    power of two (for practical reasons)

    Parameters
    ----------
    speckle : speckle image (complex field)
    filtersize : width of the gaussian filter (low pass)
    filerLevel : np.max(envelope)/filterLevel defines how big is the part
    we want to crop. If 2, we take the FWHM, for example.
    resizing: resize speckle crop to power of two size 
    (might be useful at some point)

    Returns
    -------
    roi : speckle after crops and resizing (if wanted)

    '''
    from PIL import Image #for resizing (if wanted)
    #Calculate intensity,amplitude, and phase of the field
    intensity = np.abs(speckle)**2
    amplitude = np.abs(speckle)
    phase = np.angle(speckle)
    
    #Generate low-pass filter
    fftfilter = buildGauss(speckle.shape[0], sigma = (filtersize,filtersize),
                    center = (int(speckle.shape[0]/2),int(speckle.shape[0]/2)),
                    phi = 0)
    #Low-pass filter the speckle (calculate envelope)
    speckle_filt = filt_fourier(intensity,fftfilter)
    #Calculate width of the envelope. Half maximum of the central profile
    #Calculate profile
    profile = speckle_filt[int(speckle.shape[0]/2),:]
    #Shift so the profile crosses zero at the max/filterLevel
    profile_shift = profile - np.max(profile)/filterLevel
    #find negative values
    indexes = np.where(profile_shift < 0)[0]
    #find where whe start to have positive values (jump in the indexes)
    jump = np.where(indexes[1:]-indexes[0:-1:]>1)[0]
    #find indexes in the profile that define the peak (start/end)
    start = indexes[jump[0]]
    end = indexes[jump[0]+1]
    #Crop the speckle according to the FWHM
    speckle_crop_amp = amplitude[start:end,start:end]
    speckle_crop_phase = phase[start:end,start:end]
    if resizing == True:
        #Calculate size of the new images
        size = speckle_crop_amp.shape[0]
        #Find next power of two
        new_size = 1<<size.bit_length()
        #Resize to next power of two
        speckle_crop_amp = Image.fromarray(speckle_crop_amp).resize((new_size,new_size),
                                                                    resample=Image.NEAREST)
        speckle_crop_amp = np.array(speckle_crop_amp)
        speckle_crop_phase = Image.fromarray(speckle_crop_phase).resize((new_size,new_size),
                                                                        resample=Image.NEAREST)
        speckle_crop_phase = np.array(speckle_crop_phase)
        pass
    #Merge amplitude and phase into field
    roi = speckle_crop_amp * np.exp(1j * speckle_crop_phase)

    return roi

def del_black_borders(img):
    '''
    del_black_borders trims the black borders of an image 
    (for example after masking it with a circular aperture with radius <1)                                                        )

    Parameters
    ----------
    img : image with some black borders

    Returns
    -------
    img : cropped version of the image (without the borders)

    '''
    #Remove top rows with all elements equal to zero
    while np.sum(img[0,:]) == 0:
        img = np.delete(img, 0, axis = 0)
    #Remove bottom rows with all elements equal to zero
    while np.sum(img[-1,:]) == 0:
        img = np.delete(img, -1, axis = 0)
    #Remove left columns with all elements equal to zero
    while np.sum(img[:,0]) == 0:
        img = np.delete(img, 0, axis = 1)
    #Remove right columns with all elements equal to zero
    while np.sum(img[:,-1]) == 0:
        img = np.delete(img, -1, axis = 1)
    
    return img

def norm_dimension(matrix, dim, mode = 'energy'):
    '''
    norm_dimension normalizes a matrix in one dimension (its rows or columns)
    independently (row-by-row / col-by-col). Each row/col has norm = 1 or 
    each row/col goes between 0 and 1

    Parameters
    ----------
    matrix : Input matrix. To normalize
    dim : Dimension to normalize
    mode : normalize so all the rows/cols have same energy or are between 
            0 (min) and 1 (max)
            values: 'energy' (default)
                    '0to1'
    Returns
    -------
    matrix_norm : Normalized matrix
    '''
    matrix_norm = np.zeros(matrix.shape)
    if dim == 0:
        for idx in range(0,matrix.shape[dim]):
            if mode == 'energy':
                norm = np.linalg.norm(matrix[idx,:])
                matrix_norm[idx,:] = matrix[idx,:]/norm
            elif mode == '0to1':
                matrix_norm[idx,:] = (matrix[idx,:] - np.min(matrix[idx,:])) / (np.max(matrix[idx,:]) -
                                                np.min(matrix[idx,:] + 1e-12))
            else:
                print('wrong normalization mode, return input matrix')
                return matrix
        return matrix_norm
    elif dim == 1:
        for idx in range(0,matrix.shape[dim]):
            if mode == 'energy':
                norm = np.linalg.norm(matrix[:,idx])
                matrix_norm[:,idx] = matrix[:,idx]/norm
            elif mode == '0to1':
                matrix_norm[:,idx] = (matrix[:,idx] - np.min(matrix[:,idx])) / (np.max(matrix[:,idx]) - 
                                                np.min(matrix[:,idx] + 1e-12))
            else:
                print('wrong normalization mode, return input matrix')
                return matrix
        return matrix_norm
    else:
        print('wrong dimension, return input matrix')
        return matrix
    pass
