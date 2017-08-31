This package contains the code that implements submission 0205

The app is built with a combination of MATLAB (for core computation) and C++ code (for the UI).
Source C++ with MS VS C++ project files is in the glidviewer folder.  A precompiled binary is provided with the package.

Requirements:

- MS Windows (Windows 7/8/10)
- MATLAB (>2016b)
- A GLSL 3.3 compatible GPU.
- The OpenGL UI (glidviewer.exe)

To run the software:

1. Start MATLAB
2. cd to the code folder
3. Call glid_main.m within MATLAB. This will open the main GUI, and load the data for the giraffe shape

The User Interface (main options):

For deformation, user edits the p2p constraint, i.e. 
    moving the p2p target by dragging any p2p constraint,  adding constraints by left clicking on the shape,  removing constraints by right click

- Clear P2P: remove all the p2p constraints.

2. Deformer widget


3 How to compile the binaries
** cuHarmonic.mexw64, GPU solver based on CUDA 
*** cub library
https://nvlabs.github.io/cub/

**  
The following libraries are needed to compile the OpenGL GUI  glidviewer.exe
*** Eigen
http://eigen.tuxfamily.org/index.php?title=Main_Page

*** AntTweakBar
http://anttweakbar.sourceforge.net/doc/

*** FreeGLUT
http://freeglut.sourceforge.net/


