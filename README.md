# XM: Build Rome With Convex Optimization

## STEP 1: Decide what you need
- If you already have the observation of 3D landmarks in each camera frame, you can directly pass the view-graph and observations using `./example/2_test_creatematrix.py`
  - If you found the result is not good, that is because the observation have so much noise (almost always solver will converge to global optimal, but the quality of observation indeed influence accuracy). You can refer to `.py`  for help.
- If have the images, intrinsics of cameras and corresponding depth information, you will need to install [COLMAP](https://colmap.github.io/) and [GLOMAP](https://github.com/colmap/glomap).
- If you only have images and intrinsics, you will also need depth model. Here we use [Unidepth](https://github.com/lpiccinelli-eth/UniDepth).
- If you do not have intrinsics: TODO. We are working on this right now.

## STEP 2: Installation

### Install Python environment
After clone the XM repo, make sure you are in the root path of XM folder, and run these in terminal:
```
conda create -n XM python=3.10 
conda activate XM
pip install -r requirements.txt
```

### CMake XM main solver
Directly run
```
cmake -B build .
cmake --build build
```
Note that your terminal should under the XM environment.

### Install Ceres, COLMAP and GLOMAP (CHECK WHETHER YOU NEED IT)

This part should be replaced in our final release, but for now you will need to build them as a component for our pipline.

According to [Ceres](http://ceres-solver.org/), [GLOMAP](https://github.com/colmap/glomap) and [COLMAP](https://colmap.github.io/install.html#build-from-source), you should first build Ceres and pyceres.

**Though you can install pyceres and pycolmap through `pip`, we highly recommand build from source because it support CUDA.**

#### Ceres and pyceres.

Build Ceres [from source code](http://ceres-solver.org/installation.html), and build pyceres [from source](https://github.com/cvg/pyceres) in `XM` environment.

#### COLMAP and pycolmap

Build COLMAP [from source code](https://colmap.github.io/install.html#installation), and build pycolmap [from source](https://colmap.github.io/pycolmap/index.html) in `XM` environment.

#### GLOMAP

We modified a bit on GLOMAP to fit our pipline, so you can directly build from our repository. Note GLOMAP needs COLMAP.

Run the following in root path:
```
cd deps/glomap/
mkdir build
cd build
cmake .. -GNinja
ninja && sudo ninja install
cd ../../../
```

## STEP 3: Check examples


