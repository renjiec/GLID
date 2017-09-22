This package contains the code that implements the following paper,

Renjie Chen and Ofir Weber.
GPU-Accelerated Locally Injective Shape Deformation.
ACM Transactions on Graphics, 36(6) (SIGGRAPH Asia 2017)

# What does the code contain.
The app is built with a combination of MATLAB (for core computation), C++ code (for the UI) and mex/CUDA code (for GPU accelerated optimization).
The C++ source code for the OpenGL UI with MS Visual Studio C++ project files is in the glidviewer folder.  
The mex/CUDA source code for the GPU accelerated optimization is in the cuHarmonic folder.
Precompiled binary for the UI and mex/CUDA are provided with the package.

# Requirements:
- MS Windows (Windows 7/8/10)
- MATLAB (>2016b)
- A GLSL 3.3 compatible GPU.
- The OpenGL UI (glidviewer.exe)
- CUDA (Compute Capability > 3.5)

# To run the software:
1. Start MATLAB
2. cd to the code folder
3. Call glid_main.m within MATLAB. This will automatically open the main GUI, and load the rex shape

### The User Interface (main options):
4. For deformation, the p2p constraint can be edited by
    * adding P2P constraints by left clicking on the shape
    * moving the p2p target by dragging any p2p constraint
    * removing constraints by right clicking the p2p
5. GLID Deformer widget
    * Energy, isometric energy for the optimization
    * Solver, including mesh based AQP and SLIM, and harmonic subspace based Gradient Descent, LBFGS, Newton etc.
    * #samples, number of samples on the boundary for the boundary integral approximation
    * energy param, the paramter s for Exp Symmetric Dirichlet and AMIPS energies
    * Reset Shape, reset the shape to its original state (identity mapping)
    * Pause, paurse the iteration
    * Clear P2P: remove all the p2p constraints.

# How to compile the binaries
The following libraries are needed to compile the code
1. OpenGL GUI (glidviewer.exe) 
* Eigen
http://eigen.tuxfamily.org
* AntTweakBar
http://anttweakbar.sourceforge.net
* FreeGLUT
http://freeglut.sourceforge.net

2. GPU-accelerated solver (cuHarmonic.mexw64)
* cub library
https://nvlabs.github.io/cub/

