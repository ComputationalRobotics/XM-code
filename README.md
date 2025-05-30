# XM: Build Rome With Convex Optimization

## [Website](https://computationalrobotics.seas.harvard.edu/XM/)|[Paper](https://arxiv.org/abs/2502.04640)

## About

XM is a scalable and initialization-free solver
for global bundle adjustment, leveraging learned depth and
convex optimization. This repositary implement XM and its whole structure from motion (SfM) pipeline XM-SfM, achieve huge speed up compare to existing solver.

## News

- [ ] Enable joint estimation on camera intrinsics.
- [ ] Speed up on preprocess part.
- [x] `04.23.2025`: RSS camera ready.
- [x] `03.12.2025`: Release beta version.


## STEP 1: Decide what to build
- If you already have the observation of 3D landmarks in each camera frame, you can directly pass the view-graph and observations to XM solver. See [example2](./2_test_creatematrix.py)
  - If you found the result is not good, that is because the observation have too much noise (solver will converge to global optimal, but the quality of observation indeed influence accuracy). You can refer to [example4](./4_test_unidepth.py) and [example5](./5_test_ceres.py) to use XM $^2$ and Ceres refinement. More details can refer to our paper.
- If you have images, intrinsics of cameras and corresponding depth map, you will need to install [COLMAP](https://colmap.github.io/) and [GLOMAP](https://github.com/colmap/glomap) to match corresponding feature and create view-graph.
- If you only have images and intrinsics, you will also need to install depth model to estimate depth map. Here we use [Unidepth](https://github.com/lpiccinelli-eth/UniDepth).
- If you do not have intrinsics: TODO. We are working on this right now.

## STEP 2: Resources

Before installation you need to download the test dataset from [Google Drive for Datasets](https://drive.google.com/drive/folders/13_2mcKGKVU0ibWck2n4ajUrN2MaDfR7y?usp=share_link). The `SIMPLE1` & `SIMPLE2` datasets are binary files that can directly sent to our solver. 
And `SIMPLE3` & `SIMPLE4` are image datasets. There are also datasets that you can put in `assets/Experiment/` folder to check the accuracy in paper. All datasets should be under `assets` folder, e.g. `assets/SIMPLE1` or `assets/Experiment/BAL`

## STEP 3: Installation

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
Note that your terminal should under the XM environment. You can now run [example1](./1_test_solve.py) and [example2](./2_test_creatematrix.py).

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

### Install TEASER++

For those who what to try experiment in our XM paper, you need to install [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) to calculate accuracy. Otherwise you can comment the last part of our code.

Now you can run [example3](./3_test_colmap_glomap.py)

### Install Depth Estimation Model

Our choice is [Unidepth](https://github.com/lpiccinelli-eth/UniDepth), but you may change to you custom one.

To build Unidepth directly run this:

```
cd deps/
git clone git@github.com:lpiccinelli-eth/UniDepth.git
cd UniDepth
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

Now you can run [example4](./4_test_unidepth.py) and [example5](./5_test_ceres.py).

## STEP 4: Check examples

We recommend that you read examples 1 through 5 in order.

### Example 1

This is purely XM solver, the input is the $Q$ matrix as detailed in our paper, the output is the rotation and scale of each camera.

### Example 2

Before XM solver, we add codes about how to build the $Q$ matrix from 3D observations in each frame. The input is the view-graph and 3D observation, details can be found in the original Scaled Bundle Adjustment (SBA) formulation in paper.

### Example 3

Now we add COLMAP and GLOMAP to match features and build view-graph, but use ground truth depth tp lift 2D features to 3D.

### Example 4

We add Unidepth to estimate depth information instead of ground truth depth. We also add XM $^2$ (basically run XM once and filter outliers and run again) to improve accuracy.

### Example 5

If you still find the result not accurate enough, try to run Ceres after XM. Note this is only needed when you 2D matching is accurate but you 3D estimation is bad.

## Option 1: Run the experiment in paper

If you want to run the experiment in paper, please also install [TEASER](https://github.com/MIT-SPARK/TEASER-plusplus). Note the run time can vary a lot for different GPUs. Please download dataset from [Google Drive](https://drive.google.com/drive/folders/13_2mcKGKVU0ibWck2n4ajUrN2MaDfR7y?usp=sharing) and uncompress it in `assets/Experiment` folder. We test several datasets in XM paper:

[BAL](https://grail.cs.washington.edu/projects/bal/) datasets: Please use our pre-processed data since we down-sampled from the original datasets.

[Replica](https://github.com/cvg/nice-slam) datasets: download from nice-slam repository, contains ground truth pose and depth.

[IMC](https://www.cs.ubc.ca/research/image-matching-challenge/current/) and [MipNeRF](https://jonbarron.info/mipnerf360/) datasets: contains images and ground truth pose

[TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) datasets: images and ground truth in given time stamps.




