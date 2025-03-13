# XM: Build Rome With Convex Optimization

## [Website](https://computationalrobotics.seas.harvard.edu/XM/)|[Paper](https://computationalrobotics.seas.harvard.edu/XM/static/XM.pdf)|[Arxiv](https://arxiv.org/abs/2502.04640)

## About

XM is a scalable and initialization-free solver
for global bundle adjustment, leveraging learned depth and
convex optimization. This repositary implement XM and its whole structure from motion (SfM) pipeline XM-SfM, achieve huge speed up compare to existing solver.

## News

- [ ] Enable joint estimation on camera intrinsics.
- [ ] Speed up on preprocess part.
- [x] `12.03.2025`: Release beta version.


## STEP 1: Decide what to build
- If you already have the observation of 3D landmarks in each camera frame, you can directly pass the view-graph and observations to XM solver. See [example2](./2_test_creatematrix.py)
  - If you found the result is not good, that is because the observation have so much noise (solver will converge to global optimal, but the quality of observation indeed influence accuracy). You can refer to [example4](./4_test_unidepth.py) and [example5](./5_test_ceres.py) to use XM $^2$ and Ceres refinement. More detials can refer to our paper.
- If you have images, intrinsics of cameras and corresponding depth map, you will need to install [COLMAP](https://colmap.github.io/) and [GLOMAP](https://github.com/colmap/glomap) to match corresponding feature and create view-graph.
- If you only have images and intrinsics, you will also need to install depth model to estimate depth map. Here we use [Unidepth](https://github.com/lpiccinelli-eth/UniDepth).
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
cd XM
cmake -B build .
cmake --build build
cd ..
```
Note that your terminal should under the XM environment.

### Install Ceres, COLMAP and GLOMAP (CHECK WHETHER YOU NEED IT)

This part should be replaced in our final release, but for now you will need to build them as a component for our pipeline.

According to [Ceres](http://ceres-solver.org/), [GLOMAP](https://github.com/colmap/glomap) and [COLMAP](https://colmap.github.io/install.html#build-from-source), you should first build Ceres and pyceres. You can install them into the `/deps/` folder together with GLOMAP.

**Though you can install pyceres and pycolmap through `pip`, we highly recommand build from source because it support CUDA.**

#### Ceres and pyceres.

Build Ceres [from source code](http://ceres-solver.org/installation.html), and build pyceres [from source](https://github.com/cvg/pyceres) in `XM` environment.

#### COLMAP and pycolmap

Build COLMAP [from source code](https://colmap.github.io/install.html#installation), and build pycolmap [from source](https://colmap.github.io/pycolmap/index.html) in `XM` environment.

#### GLOMAP

We modified a bit on GLOMAP to fit our pipline, so you can directly build from our repository. Note GLOMAP needs COLMAP.

Run the following in root path of XM:
```
cd deps/glomap/
mkdir build
cd build
cmake .. -GNinja
ninja && sudo ninja install
cd ../../../
```

### Install Depth Estimation Model

Our choice is [Unidepth](https://github.com/lpiccinelli-eth/UniDepth), but you may change to you custom one.

To build Unidepth directly run this:

```
cd deps/
git clone git@github.com:lpiccinelli-eth/UniDepth.git
# Change to your own CUDA version
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu124
```
<details>
<summary>You may encouter the same issue as me:</summary>

- If pytorch3d cannot build, please comment the line about pytorch in `Unidepth/requirement.txt` and retry. After successfully installing other dependence, build pytorch3d again.

- If `name 'warnings' is not defined`, you may need to add `import warnings` in the corresponding file.

- It will show some warning about timm, but that do not hurt.

- If loded together with `XM` or `pycolmap`, `pyceres` using `import`, UniDepth must be load before them.
</details>

## STEP 3: Check examples

### Example 1




