## XM: Build Rome With Convex Optimization

### STEP 1: Decide what you need
- If you already have the observation of 3D landmarks in each camera frame, you can directly pass the view-graph and observations using `./example/2_test_creatematrix.py`
  - If you found the result is not good, that is because the observation have so much noise (almost always solver will converge to global optimal, but the quality of observation indeed influence accuracy). You can refer to `.py`  for help.
- If have the images, intrinsics of cameras and corresponding depth information, you will need to install [COLMAP](https://colmap.github.io/) and [GLOMAP](https://github.com/colmap/glomap).
- If you only have images and intrinsics, you will also need depth model. Here we use [Unidepth](https://github.com/lpiccinelli-eth/UniDepth).
- If you do not have intrinsics: TODO. We are working on this right now.

### STEP 2: Installation

#### Install Python environment
After clone the XM repo, make sure you are in the root path of XM folder, and run these in terminal:
```
conda create -n XM python=3.10 
conda activate XM
pip install -r requirements.txt
```

#### CMake XM main solver
Directly run
```
cmake -B build .
cmake --build build
```
Note that your terminal should under the XM environment.

#### Install COLMAP and GLOMAP (CHECK WHETHER YOU NEED IT)

This part should be replaced in our final release, but for now you will need to build them as a component for our pipline.

According to [GLOMAP](https://github.com/colmap/glomap) and [COLMAP](https://colmap.github.io/install.html#build-from-source), you should first install COLMAP dependencies:

```
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
```
and CUDA dependencies (We highly recommend this for big datasets):

```
sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc
```
Then build GLOMAP:

```
cd deps/glomap
mkdir build
cd build
cmake .. -GNinja
ninja && sudo ninja install
cd ../../../
```

And build [pycolmap](https://colmap.github.io/pycolmap/index.html)