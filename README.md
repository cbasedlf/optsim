# optsim
 Tools for Optical Simulations

This started as a small library of optical propagation functions during the first Covid confinement, as a way to learn using Python. It covered the basic stuff: Fresnel propagation, Fraunhoffer, angular spectral method, etc. However, with time it started growing and I included more and more functions related to the work I was doing at that time in the lab. 

I have been using this for a good couple of years and it has simplified doing simulations a lot. Now now only allows to do propagations, but can also simulate speckle patterns coming from a thin scattering medium, calculate wavefronts from images obtained with Shack-Hartman wavefront sensors (and decompose wavefronts in Zernike polynomials), simulate some very basic neuronal activities, do some transformation on light field images, simulate wavefronts arising from point sources, and also some quality of life functions to plot images and generating animations using matplotlib.

I always worked on Spyder due to its similarities with Matlab (that's the software I was using when I started doing optics), and you can even see that the way I coded stuff is super inspired by how the same ideas would have been implemented on Matlab. This will make it easy for some people walking the same path I did, but probably python enthusiasts will go crazy over the code. In my defense, I will say that some of these functions were my introduction to the language, and for sure nowadays I would implement them in different ways. Also, it works!

The library itself is relatively well commented, so anyone familiar with the physical phenomena in play should easily grasp what the code is doing. However, I will try to add a brief description of each function in this document. Also, I have added some small snippets of code with simple examples of the stuff you can do with these functions


Speckle pattern evolving along the optical axis:
![speckle_evolution](https://user-images.githubusercontent.com/19323057/192783633-74506261-36b4-44ed-948d-72d7c3392964.gif)
