# optsim: tools for optical simulations

This started as a small library of optical propagation functions during the first Covid confinement, as a way to learn using Python. It covered the basic stuff: Fresnel propagation, Fraunhoffer, angular spectrum method, etc. However, with time it started growing and I included more and more functions related to the work I was doing at that time in the lab. 

I have been using this for a good couple of years and it has simplified doing simulations a lot. Now not only it allows to do propagations, but can also be used to simulate speckle patterns coming from a thin scattering medium, calculate wavefronts from images obtained with Shack-Hartman wavefront sensors (and decompose wavefronts in Zernike polynomials), simulate some very basic neuronal activities, do some transformations on light field images, simulate wavefronts arising from point sources, and also some quality of life functions to plot images and generating animations using matplotlib.

I always worked on Spyder due to its similarities with Matlab (that's the software I was using when I started doing optics), and you can even see that the way I coded stuff is super inspired by how the same ideas would have been implemented on Matlab. This will make it easy for some people walking the same path I did, but probably python enthusiasts will go crazy over the code. In my defense, I will say that some of these functions were my introduction to the language, and for sure nowadays I would implement them in different ways. Also, it works!

The library itself is relatively well commented, so anyone familiar with the physical phenomena in play should easily grasp what the code is doing. However, I will try to add a brief description of each function in this document. Also, I have added some small snippets of code with simple examples of the stuff you can do with these functions

It uses the following libraries:
Numpy, Matplotlib, OpenCV, PIL, Skimage, Scipy

Images are usually treated as real (complex when working with optical fields) 2D numpy arrays (i.e., real matrices). In the same spirit, phase objects (lenses, scattering media) and filters are also built as matrices. No Pytorch implementation for now (I never really had the need to use Pytorch with most of these functions), but should be relatively easy to do (send a message if you are interested!).

## Install
Clone the repo, download and use this code snippet to generate a conda environment called optsim and install both optsim and the needed libraries to run it (numpy,matplotlib,etc.).
Of course, you do not need to create a new environment and you can install it in the one you tipically use. 
Once downloaded, just go to the directory where you saved the repo and run the pip install line.
```
conda create --name optsim python=3.9
conda activate optsim

pip install -e .
```

## Function descriptions:
I divided the library into six main groups: Optical propagation, Quality of life (plotting & saving images, representing complex fields, doing simple manipulations of images such as ROI selection, etc.), general optics simulation tools (build apertures, phase distributions of lenses, common image filters, compare images and measure quality), light field / wavefront sensing, neuronal activity simulation, and miscellaneous.
For a more detailed description of each function, read function_description.md and/or the comments on the code.

## Result examples:
### Example#1: Random phase mask used to simulate a thin scattering material:
![scattering_layer](https://user-images.githubusercontent.com/19323057/192787017-b31b0166-1f00-43bf-8e8b-f30189e0de98.png)
### Example#2: Speckle pattern evolving along the optical axis (you can see caustics at close distance from the diffuser at the start of the animation, and the speckle evolving as propagation occurs). Propagation made with the angular spectrum method, example in generate_speckle.py:
![speckle_evolution](https://user-images.githubusercontent.com/19323057/192783633-74506261-36b4-44ed-948d-72d7c3392964.gif)
### Example#3: Speckle generation from a point source at different lateral positions (finite memory effect), example in generate_speckle.py:
![finite_lateral](https://user-images.githubusercontent.com/19323057/193068815-6b32177d-d0df-4987-a0ba-c6f462279e6d.gif)
### Example#4: Speckle generation from plane wave illumination of a scattering layer (infinite memory effect), example in generate_speckle.py
![finite_lateral_tilt](https://user-images.githubusercontent.com/19323057/193068904-3212db15-11bb-4fd4-aadb-7b5a240a953d.gif)
