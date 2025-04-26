# change to your own path if Unidepth is installed in other places
from deps.UniDepth.unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
# Warning: THis must be the first line, otherwise pyceres, pycolmap and XM are also using CUDA
# and CUDA initialization will give you core dumped error (don't know why)

import pyceres
import pycolmap

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'XM/build/')))
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import torch
import open3d as o3d
import random
import pickle
import itertools
from scipy.stats import trim_mean

import XM

from scipy.sparse import coo_matrix, save_npz, load_npz

from utils.readgt_replica import load_replica_gt, load_replica_camera
from utils.readgt_colmap import load_colmap_gt, load_colmap_camera
from utils.readgt_TUM import load_tum_gt, load_tum_camera
from utils.cameramath import quat2rot
from utils.checkconnection import checklandmarks
from utils.creatematrix import create_matrix
from utils.io import save_matrix_to_bin, load_matrix_from_bin
from utils.recoversolution import recover_XM
from utils.visualization import visualize_camera, visualize
from utils.ceresforXM import XM_Ceres_interface
from utils.error import ATE_TEASER_C2W

current_dir = os.path.dirname(os.path.abspath(__file__))

############################
# For experiment in paper only
# download the dataset from:
# https://drive.google.com/drive/folders/13_2mcKGKVU0ibWck2n4ajUrN2MaDfR7y?usp=sharing
# and put it in the folder './assets/Experiment/**'
############################

# # MipNerf Datasets: kitchen, room, garden
# # IMC Datasets: rome, temple, gate
# # TUM Datasets: TUM_computer_R,TUM_computer_T,TUM_desk,TUM_room
# dataset_path = os.path.abspath(os.path.join(current_dir, "./assets/Experiment/MipNerf/kitchen"))
dataset_path = os.path.abspath(os.path.join(current_dir, "./assets/SIMPLE4"))

image_dir = dataset_path + '/images'
output_path = dataset_path + '/output'
database_path = output_path + "/database.db"

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# Decide which one to run
# input: image
run_colmap = 1
# input: database
run_glomap = 1
# input: view graph + 2D observations
run_depth = 1
# input: edges and 3D observations
run_filter = 1
visualization_glomap_filter = 0

# load camera information and gt poses (if needed)
gt = load_colmap_gt(dataset_path)
gt_camera = load_colmap_camera(dataset_path)

# gt = load_tum_gt(dataset_path)
# gt_camera = load_tum_camera(dataset_path)

"""
File Purpose:
    This file processes input data consisting of images and camera intrinsic parameters.
    The processing pipeline includes the following steps:

    1. COLMAP Matching:
         - Perform feature matching across the input images using COLMAP to establish reliable correspondences.
    2. GLOMAP Indexing:
         - Index the matched features using GLOMAP for efficient retrieval and further processing.
    3. 2D to 3D Lifting:
         - Lift 2D feature observations into 3D space using the available depth information (estimated or provided).
    4. XM and XM^2 Computation:
         - Run the XM algorithm to obtain an initial solution (e.g., a 3D reconstruction or pose estimation).
    5. Ceres Refinement:
         - Refine the initial solution obtained from XM using the Ceres optimizer to improve accuracy and consistency.

Usage Note:
    Ensure that the input images and intrinsic parameters are correctly pre-processed and aligned before running this pipeline.
    The Ceres optimizer is employed after the XM step to fine-tune the results, so proper configuration of the optimizer parameters is crucial.
"""


# Run COLMAP feature extracting and matching
if run_colmap:
    if len(gt_camera) == 1:
        # single camera
        ImageReaderOptions = pycolmap.ImageReaderOptions()
        print("Camera model: ", gt_camera[1]["model"])
        params = gt_camera[1]["params"]  # e.g., np.array([600, 600, 599.5, 339.5])
        param_str = ','.join(map(str, params))
        ImageReaderOptions.camera_params = param_str
        pycolmap.extract_features(database_path, image_dir,camera_mode = pycolmap.CameraMode.SINGLE, camera_model="PINHOLE",reader_options=ImageReaderOptions)
    else:
        # if you have different cameras, may refer to COLMAP document about how to use intrinsics
        pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)

# Run GLOMAP indexing
if run_glomap:
    glomap_executable_path = "glomap"

    command = [
        glomap_executable_path,
        "mapper",
        "--database_path", database_path,  
        "--output_path", output_path+"/glomap_output",      
        "--image_path", image_dir ,
        "--TrackEstablishment.max_num_view_per_track", "1000000" 
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,  
        stderr=subprocess.PIPE,  
        text=True                
    )

    try:
        while True:
            output = process.stdout.readline()  
            if output == "" and process.poll() is not None:
                break  
            if output:
                print(output.strip())  
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        process.stdout.close()
        process.stderr.close()

    return_code = process.poll()
    if return_code == 0:
        print("GLOMAP executed successfully!")
    else:
        print(f"GLOMAP failed with return code {return_code}.")

    # after running GLOMAP will output three .txt file in the assets/tempdata/ folder

    matches = np.loadtxt(os.path.abspath(os.path.join(current_dir, "./assets/tempdata/output.txt"))) 
    File3 = pd.read_table(os.path.abspath(os.path.join(current_dir, "./assets/tempdata/filename.txt")), header=None, names=["Var1", "Var2"], sep='\s+')
    relpose = np.loadtxt(os.path.abspath(os.path.join(current_dir, "./assets/tempdata/relative_pose.txt")))

    File3 = File3.sort_values(by="Var1").reset_index(drop=True)

    max_index = File3['Var1'].max()
    filename = np.empty(max_index, dtype=object) 

    indices = File3['Var1'].to_numpy() - 1 
    values = File3['Var2'].to_numpy()       
    filename[indices] = values        

    glomap_pose = {}
    for i in range(relpose.shape[0]):
        image_id1 = int(relpose[i, 0])
        image_id2 = int(relpose[i, 1])
        R = quat2rot(relpose[i, 2], relpose[i, 3], relpose[i, 4], relpose[i, 5])
        t = relpose[i, 6:9]
        glomap_pose[(image_id1, image_id2)] = (R, t)  
    
    with open(output_path + "/glomap_pose.pkl", "wb") as file:
        pickle.dump(glomap_pose, file)
        
    np.save(output_path + "/filename.npy", filename)

    N = int(np.max(matches[:, 0]))  
    M = int(np.max(matches[:, 3]))  

    print("N:  ", N)
    print("M:  ", M)
    print("Observations:  ", matches.shape[0])

    # this edges is 1-base
    edges = matches[:, [0, 3]].copy()
    delete_observation = edges.shape[0]
    sorted_indices = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[sorted_indices]
    matches = matches[sorted_indices]
    _, unique_indices = np.unique(edges, axis=0, return_index=True)
    matches = matches[unique_indices]
    delete_observation = delete_observation - matches.shape[0]
    print("delete same observation:  ", delete_observation)  
    
    vis = coo_matrix((np.ones(matches.shape[0], dtype=int), (matches[:, 0].astype(int)-1, matches[:, 3].astype(int)-1))).tocsr()
    landmarkx = coo_matrix((matches[:, 1], (matches[:, 0].astype(int)-1, matches[:, 3].astype(int)-1))).tocsr()
    landmarky = coo_matrix((matches[:, 2], (matches[:, 0].astype(int)-1, matches[:, 3].astype(int)-1))).tocsr()

    # save the vis, landmarkx, landmarky
    save_npz(output_path + "/vis.npz", vis)
    save_npz(output_path + "/landmarkx.npz", landmarkx)
    save_npz(output_path + "/landmarky.npz", landmarky)
else:
    # load file
    if not os.path.exists(output_path + "/vis.npz"):
        print("No data found, please run the GLOMAP first.")
        exit()
    else:
        vis = load_npz(output_path + "/vis.npz")
        landmarkx = load_npz(output_path + "/landmarkx.npz")
        landmarky = load_npz(output_path + "/landmarky.npz")
        filename = np.load(output_path + "/filename.npy", allow_pickle=True)
        N = vis.shape[0]
        M = vis.shape[1]
    

# Run depth lifting, for this dataset we have ground truth depth
points_3d = np.zeros((0, 3))
weights = np.array([])
edges = np.zeros((0, 2))
rgbs = np.zeros((0, 3))
if run_depth:
    print("Torch version:", torch.__version__)
    type_ = "l"  # available types: s, b, l
    name = f"unidepth-v2-vit{type_}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    # set resolution level (only V2)
    model.resolution_level = 9

    # set interpolation mode (only V2)
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    for i in tqdm(range(N), desc="Processing images depth"):
        rgb = cv2.imread(image_dir + f"/{filename[i]}")
        # depth estimation
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
        predictions = model.infer(rgb_torch)

        depth_pred = predictions["depth"].squeeze().cpu().numpy()
        confidence_pred = predictions["confidence"].squeeze().cpu().numpy()

        if len(depth_pred.shape) == 1:
            continue
        
        col_indices = vis[i].nonzero()[1]
        if len(col_indices) == 0:
            continue
        point_frame = []
        h_image, w_image = depth_pred.shape
        u = landmarkx.getrow(i).data.astype(int)
        v = landmarky.getrow(i).data.astype(int)
        
        margin = 10
        valid_indices = (u >= margin) & (u < w_image - margin) & (v >= margin) & (v < h_image - margin)
        u = u[valid_indices]
        v = v[valid_indices]
        col_indices = col_indices[valid_indices]
        if u.shape[0] == 0:
            continue    
        
        d = depth_pred[v, u] 
        threshold = np.percentile(d, 95)

        valid_depth_indices = (d > 0) & (d < threshold)
        u = u[valid_depth_indices]
        v = v[valid_depth_indices]
        d = d[valid_depth_indices]
        col_indices = col_indices[valid_depth_indices]
        
        w = confidence_pred[v,u]
        pixel_coords = np.vstack((u, v, np.ones_like(u) * 1))
        # normalization
        camera_coords = np.linalg.inv(gt[filename[i]]["K"]) @ pixel_coords
        camera_coords = camera_coords.T * d[:, np.newaxis]
        
        # stack
        points_3d = np.vstack((points_3d, camera_coords))
        weights = np.hstack((weights, w**2))
        edges = np.vstack((edges, np.vstack((np.ones_like(u)*i, col_indices)).T))
        rgbs = np.vstack((rgbs, rgb[v,u]))
        
    landmarks = np.array(points_3d)
    weights = np.array(weights)
    edges = np.array(edges).astype(int) + 1
    rgbs = np.array(rgbs)

    np.save(output_path + "/weights.npy", weights)
    np.save(output_path + "/edges.npy", edges)
    np.save(output_path + "/landmarks.npy", landmarks)
    np.save(output_path + "/rgbs.npy", rgbs)
else:
    if os.path.exists(output_path + "/weights.npy"):
        weights = np.load(output_path + "/weights.npy")
        edges = np.load(output_path + "/edges.npy")
        landmarks = np.load(output_path + "/landmarks.npy")
        rgbs = np.load(output_path + "/rgbs.npy")
    else:
        print("No data found, please run the depth estimation first.")
        exit()


######################    
# YOUR OWN FILTER HERE
######################
if run_filter:
    with open(output_path + "/glomap_pose.pkl", "rb") as file:
        glomap_pose = pickle.load(file)
    vis = coo_matrix(((np.arange(edges.shape[0])), (edges[:,0]-1, edges[:,1]-1)), shape=(N, M)).tocsr()
    num_deleta_observation_GLOMAP_FILTER = 0
    is_outlier = np.zeros(edges.shape[0], dtype=bool)
    pairs = itertools.combinations(range(N), 2)
    error_sum = np.zeros((N, M))

    for (i, j) in tqdm(pairs, total=N*(N-1)//2, desc="Processing pairs", unit="pair"):
        # find the inersection of i and j
        R, t = glomap_pose.get((i+1, j+1), (None, None))
        if R is None:
            continue
        points_index1 = vis.getrow(i).toarray().ravel()
        points_index2 = vis.getrow(j).toarray().ravel()
        joint_points_index = np.where(np.logical_and(points_index1, points_index2))[0]

        if len(joint_points_index) < 20:
            continue

        src = landmarks[vis[i,joint_points_index].toarray().squeeze(),:].T
        dst = landmarks[vis[j,joint_points_index].toarray().squeeze(),:].T

        # estimate scale
        dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
        src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
        dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
        src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
        # delete 10% outliers
        index = (src_dis < np.percentile(src_dis, 90)) & (dst_dis < np.percentile(dst_dis, 90))
        src_n = src[:, index]
        dst_n = dst[:, index]
        dst_avg = trim_mean(dst_n, proportiontocut=0.05, axis=1)
        src_avg = trim_mean(src_n, proportiontocut=0.05, axis=1)
        
        scale1 = trim_mean(np.linalg.norm(dst_n - dst_avg.reshape(3, 1), axis=0), proportiontocut=0.05)
        scale2 = trim_mean(np.linalg.norm(src_n - src_avg.reshape(3, 1), axis=0), proportiontocut=0.05)

        src = src / scale2 * scale1
        # because we have scale, so this may differs
        
        src_noR = R @ src
        translation = trim_mean(dst - src_noR, proportiontocut=0.05, axis=1)
        
        target = src_noR + translation.reshape(3, 1)

        # relative error
        error = np.linalg.norm(target - dst, axis=0)/scale1
        error_angle = np.arccos(np.sum(target * dst, axis=0) / (np.linalg.norm(target, axis=0) * np.linalg.norm(dst, axis=0)))
        
        percentage = np.sum(error < 0.05)/error.shape[0]

        thereshold = 3 * np.median(error)
        outliers = error -  max(thereshold, np.percentile(error,95)) > 0
        inliers = ~outliers

        if  np.sum(outliers) > 0:
            # num_deleta_observation_GLOMAP_FILTER += 2 * np.sum(outliers)
            # is_outlier[vis[i, joint_points_index[outliers]].toarray().squeeze()] = True
            # is_outlier[vis[j, joint_points_index[outliers]].toarray().squeeze()] = True
            error_sum[i, joint_points_index[outliers]] += 1
            error_sum[j, joint_points_index[outliers]] += 1
            
            if visualization_glomap_filter:
            
                target_inlier = target[:,inliers]
                dst_inlier = dst[:,inliers]
                target_outlier = target[:,outliers]
                dst_outlier = dst[:,outliers]

                lines = o3d.geometry.LineSet()

                all_points = np.vstack((target.T, dst.T))
                lines.points = o3d.utility.Vector3dVector(all_points)

                num_points = target.shape[1]
                line_indices = [[i, i + num_points] for i in range(num_points)]
                lines.lines = o3d.utility.Vector2iVector(line_indices)

                lines.paint_uniform_color([0, 1, 0])  # green
                
                src_o3d = o3d.geometry.PointCloud()
                src_o3d.points = o3d.utility.Vector3dVector(target_inlier.T)
                src_o3d.paint_uniform_color([0, 0, 1])  # blue

                dst_o3d = o3d.geometry.PointCloud()
                dst_o3d.points = o3d.utility.Vector3dVector(dst_inlier.T)
                dst_o3d.paint_uniform_color([1, 0, 0])  
                
                src_O_o3d = o3d.geometry.PointCloud()
                src_O_o3d.points = o3d.utility.Vector3dVector(target_outlier.T)
                src_O_o3d.paint_uniform_color([0, 0, 0])  

                dst_O_o3d = o3d.geometry.PointCloud()
                dst_O_o3d.points = o3d.utility.Vector3dVector(dst_outlier.T)
                dst_O_o3d.paint_uniform_color([0, 0, 0]) 

                # Visualize:
                o3d.visualization.draw_geometries([src_o3d, dst_o3d ,lines, src_O_o3d, dst_O_o3d])
    for i in range(M):
        involved_index = np.where(error_sum[:,i]>0)[0]
        if len(involved_index) > 0:
            max_involved = np.max(error_sum[involved_index,i])
            outliers = np.where(error_sum[involved_index,i] > 0)[0]
            is_outlier[vis[involved_index[outliers], i].toarray().squeeze()] = True
            # print("landmark ", i, "error sum: ", error_sum[np.where(error_sum[:,i]>0),i], "total frames", vis.getcol(i).nnz) 
    edges = edges[~is_outlier]
    weights = weights[~is_outlier]
    landmarks = landmarks[~is_outlier]
    rgbs = rgbs[~is_outlier]
    N = edges[:,0].max()
    M = edges[:,1].max()

    print("Total remain observations after glomap pose:", edges.shape[0])
    print("Total delete observations after glomap pose:", np.sum(is_outlier))
    # save data
    np.save(output_path + "/weights_filter.npy", weights)
    np.save(output_path + "/edges_filter.npy", edges)
    np.save(output_path + "/landmarks_filter.npy", landmarks)
    np.save(output_path + "/rgbs_filter.npy", rgbs) 
else:
    # judge if they exist
    if os.path.exists(output_path + "/weights_filter.npy") and not run_depth:
        weights = np.load(output_path + "/weights_filter.npy")
        edges = np.load(output_path + "/edges_filter.npy")
        landmarks = np.load(output_path + "/landmarks_filter.npy")
        rgbs = np.load(output_path + "/rgbs_filter.npy")
        N = edges[:,0].max()
        M = edges[:,1].max()

# # debug
# j = 0
# for i in range(N):
#     vis = coo_matrix(((np.arange(edges.shape[0])), (edges[:,0]-1, edges[:,1]-1)), shape=(N, M)).tocsr()
#     points_index1 = vis.getrow(i).toarray().ravel()
#     points_index2 = vis.getrow(j).toarray().ravel()
#     joint_points_index = np.where(np.logical_and(points_index1, points_index2))[0]

#     if len(joint_points_index) < 20:
#         continue

#     src = landmarks[vis[i,joint_points_index].toarray().squeeze(),:]
#     dst = landmarks[vis[j,joint_points_index].toarray().squeeze(),:]

#     pcd_src = o3d.geometry.PointCloud()
#     pcd_src.points = o3d.utility.Vector3dVector(src)
#     pcd_src.paint_uniform_color([0, 0, 1])
#     pcd_dst = o3d.geometry.PointCloud()
#     pcd_dst.points = o3d.utility.Vector3dVector(dst)
#     pcd_dst.paint_uniform_color([1, 0, 0])

#     all_points = np.vstack((src, dst))
#     num_points = src.shape[0]

#     lines = np.array([[i, i + num_points] for i in range(num_points)])

#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(all_points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector([[1, 0, 1] for _ in range(num_points)])

#     o3d.visualization.draw_geometries([pcd_src,pcd_dst,line_set])

# send it to XM
# check is the view-graph is connected
edges, landmarks, weights, rgbs, indices = checklandmarks(edges, landmarks, weights, rgbs, N, M)
indices_all = indices.copy()

create_matrix(weights, edges, landmarks, output_path)
lam = edges.shape[0] / N
# sometimes change it to 1e-3
XM.solve(output_path, 5, 1e-1, lam, 1000)

# visualize the camera poses
Abar,_ = load_matrix_from_bin(output_path + '/Abar.bin')
R,_ = load_matrix_from_bin(output_path + '/R.bin')
s,_ = load_matrix_from_bin(output_path + '/s.bin')
Q,_ = load_matrix_from_bin(output_path + '/Q.bin')        

# recover p and t
R_real, s_real, p_est, t_est = recover_XM(Q, R, s, Abar, lam)

N = s_real.shape[0]
M = p_est.shape[1]

# XM^2
# some times the result is not good, so we need to remove some outliers
# this is basically a robust outlier removal, you delete landmarks that has the largest error

src_idx = edges[:, 0] - 1  # Convert from 1-based to 0-based indexing
dst_idx = edges[:, 1] - 1

R_real = R_real.reshape(3, N, 3).transpose(1, 0, 2)  # Shape: (M, 3, 3)

# Extract the correct 3x3 rotation matrices
R_matrices = R_real[src_idx]  # Shape: (N, 3, 3)

# Compute transformed landmarks
landmarks_transformed = (s_real[src_idx, None] * np.einsum('nij,nj->ni', R_matrices, landmarks)) + t_est[:, src_idx].T

diff = (p_est[:, dst_idx].T - landmarks_transformed)  # Ensure shapes match
squared_distances = np.sum(diff**2, axis=1)

error = weights * squared_distances

print("sum of error: ", np.sum(error))

threshold = np.percentile(error, 90)  # 90th percentile

indices_to_remove = np.where(error > threshold)[0] 

# # DEBUG on this
# for i in range(N):
#     indices_i_frame = np.where(edges[:, 0] == i)[0]
#     # find the index where overlap
#     overlap_mask = np.in1d(indices_i_frame, indices_to_remove)
#     overlap = np.where(overlap_mask)[0]

#     world_points = p_est[:, edges[indices_i_frame, 1] - 1].T
#     camera_points = np.einsum('ij,nj->ni', R_real[i-1], landmarks[indices_i_frame,:]) + t_est[:, i-1]
#     world_points_inliers = world_points[np.setdiff1d(np.arange(world_points.shape[0]), overlap)]
#     camera_points_inliers = camera_points[np.setdiff1d(np.arange(camera_points.shape[0]), overlap)]
#     world_points_outliers = world_points[overlap]
#     camera_points_outliers = camera_points[overlap]
#     # show with open3d
#     o3d_world = o3d.geometry.PointCloud()
#     o3d_world.points = o3d.utility.Vector3dVector(world_points_inliers)
#     o3d_world.paint_uniform_color([0, 0, 1])

#     o3d_camera = o3d.geometry.PointCloud()
#     o3d_camera.points = o3d.utility.Vector3dVector(camera_points_inliers)
#     o3d_camera.paint_uniform_color([1, 0, 0])

#     o3d_world_outliers = o3d.geometry.PointCloud()
#     o3d_world_outliers.points = o3d.utility.Vector3dVector(world_points_outliers)
#     o3d_world_outliers.paint_uniform_color([0, 1, 0])

#     o3d_camera_outliers = o3d.geometry.PointCloud()
#     o3d_camera_outliers.points = o3d.utility.Vector3dVector(camera_points_outliers)
#     o3d_camera_outliers.paint_uniform_color([1, 1, 0])

#     # draw lines
#     all_points = np.vstack((world_points_outliers, camera_points_outliers))
#     num_outliers = world_points_outliers.shape[0]

#     lines = np.array([[i, i + num_outliers] for i in range(num_outliers)])

#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(all_points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector([[1, 0, 1] for _ in range(num_outliers)])

#     o3d.visualization.draw_geometries([o3d_world, o3d_camera, o3d_world_outliers, o3d_camera_outliers, line_set])


edges = np.delete(edges, indices_to_remove, axis=0)  
weights = np.delete(weights, indices_to_remove)
rgbs = np.delete(rgbs, indices_to_remove, axis=0)
landmarks = np.delete(landmarks, indices_to_remove, axis=0)

# second run
edges, landmarks, weights, rgbs, indices = checklandmarks(edges, landmarks, weights, rgbs, N, M)
N_old = np.where(indices_all > -1)[0].shape[0]
indices_all_copy = indices_all.copy()
for i in range(N_old):
    indices_all[np.where(indices_all_copy == i)[0]] = indices[i]

create_matrix(weights, edges, landmarks, output_path)
lam = 0.0
XM.solve_rank3(output_path, 3, 1e-1, lam, 1000)
s,_ = load_matrix_from_bin(output_path + '/s.bin')
s_avg = np.mean(s[1:])
s_std = np.std(s[1:])
# decide whether add regularization term
if np.abs(s_avg - 1)> 2 * s_std or np.sum(s < 0.1) > 10:
    print("s is too small, run again")
    lam = edges.shape[0] / N
    XM.solve(output_path, 5, 1e-1, lam, 1000)
else:
    print("s is good")
    XM.solve(output_path, 5, 1e-1, lam, 1000)

Abar,_ = load_matrix_from_bin(output_path + '/Abar.bin')
R,_ = load_matrix_from_bin(output_path + '/R.bin')
s,_ = load_matrix_from_bin(output_path + '/s.bin')
Q,_ = load_matrix_from_bin(output_path + '/Q.bin')        

# recover p and t
R_real, s_real, p_est, t_est = recover_XM(Q, R, s, Abar, lam)

N = s_real.shape[0]
M = p_est.shape[1]

# If you still do not have a good result, consider using Ceres to refine using 2D observations.
# But make sure your 2D observation is good enough -- otherwise this won't help

landmarks_2D = landmarks[:, :2] / landmarks[:, 2, None]
nan_mask = np.isnan(landmarks_2D).any(axis=1)

R_real, t_est, p_est, ceres_time, ceres_cost = XM_Ceres_interface(edges[~nan_mask], landmarks_2D[~nan_mask], R_real, t_est, p_est)

sR_real = np.zeros((3, 3*N))
for i in range(N):
    sR_real[:,3*i:3*i+3] = s_real[i] * R_real[:,3*i:3*i+3]
    
# Compute ybar_est
ybar_est = Abar @ sR_real.T

# Add a column of zeros to the left
y_est = np.hstack((np.zeros((3, 1)), ybar_est.T))

p_est = y_est[:, N:N + M]

extrinsics = []
for i in range(N):
    # extrinsic is 4 * 4
    # extrinsic is world 2 camera, while R_real is camera 2 world
    extrinsics.append(np.vstack((np.hstack((R_real[:,3*i:3*i+3].T, -R_real[:,3*i:3*i+3].T @ t_est[:,i].reshape([3,1]))), np.array([0, 0, 0, 1]))))

indices = edges[:, 1] - 1  
mean_rgbs = np.zeros((M, 3))

np.add.at(mean_rgbs, indices, rgbs)

counts = np.bincount(indices, minlength=M)[:, None] 
mean_rgbs /= counts  

mean_rgbs = mean_rgbs[:, [2, 1, 0]]

# # visualize all
visualize(extrinsics, p_est.T, mean_rgbs/255.0)

# accuracy
# comment this and the corresponding import if you do not need this
data_RPE_R = []
data_RPE_T = []
data_ATE_R = []
data_ATE_T = []
t_gt = np.zeros((3, N))
R_gt = np.zeros((3, 3 * N))
for i in range(N):
    i_index = np.where(indices_all == i)[0][0]
    t_gt[:,i] = gt[filename[i_index]]["t"]  
    R_gt[:,3*i:3*i+3] = gt[filename[i_index]]["R"]

avg_t_gt = np.mean(t_gt,axis=1)
cov_t_gt = np.mean(np.linalg.norm(t_gt - avg_t_gt.reshape(3,1),axis=0))

s_g, R_g, t_g = ATE_TEASER_C2W(R_real,t_est,R_gt,t_gt)

ATE_R = np.zeros(N)
ATE_T = np.zeros(N)

for i in range(N):
    cosvalue = (np.trace(R_g @ R_real[:,3*i:3*i+3] @ R_gt[:,3*i:3*i+3])-1)/2
    ATE_R[i] = np.abs(np.arccos(min(max(cosvalue,-1),1)))
    ATE_T[i] = np.linalg.norm((s_g * R_g @ t_est[:,i] + t_g.flatten())-R_gt[:,3*i:3*i+3].T @ (-t_gt[:,i]))
    
RPE_R = []
RPE_t = []
for i in range(N):
    if N > 1000:
        for j in random.sample(list(np.arange(N)), 100):
            cosvalue = (np.trace(R_gt[:,3*i:3*i+3] @ R_gt[:,3*j:3*j+3].T @ R_real[:,3*j:3*j+3].T @  R_real[:,3*i:3*i+3])-1)/2
            RPE_R.append(np.abs(np.arccos(min(max(cosvalue,-1),1))))
            RPE_t.append(np.linalg.norm(- R_gt[:,3*i:3*i+3].T @ t_gt[:,i] + R_gt[:,3*j:3*j+3].T @ t_gt[:,j] - s_g * R_g @ (t_est[:,i] - t_est[:,j])))
    else:
        for j in range(i):
            cosvalue = (np.trace(R_gt[:,3*i:3*i+3] @ R_gt[:,3*j:3*j+3].T @ R_real[:,3*j:3*j+3].T @  R_real[:,3*i:3*i+3])-1)/2
            RPE_R.append(np.abs(np.arccos(min(max(cosvalue,-1),1))))
            RPE_t.append(np.linalg.norm(- R_gt[:,3*i:3*i+3].T @ t_gt[:,i] + R_gt[:,3*j:3*j+3].T @ t_gt[:,j] - s_g * R_g @ (t_est[:,i] - t_est[:,j])))
    
print('RPE-R: ', np.median(RPE_R),'RPE-T: ', np.median(RPE_t)/cov_t_gt,'ATE-R: ', np.median(ATE_R),'ATE-T: ', np.median(ATE_T)/cov_t_gt)
data_RPE_R.append(np.median(RPE_R))   
data_RPE_T.append(np.median(RPE_t)/cov_t_gt)
data_ATE_R.append(np.median(ATE_R))
data_ATE_T.append(np.median(ATE_T)/cov_t_gt)


print("\stackon{$",
    np.round(np.median(ATE_T)/cov_t_gt,3), "$}{$", np.round(np.degrees(np.median(ATE_R)),3), "^{\circ}$} & \stackon{$",
    np.round(np.median(RPE_t)/cov_t_gt,3), "$}{$", np.round(np.degrees(np.median(RPE_R)),3), "^{\circ}$}"
)
output_str = (
    "\stackon{$"
    f"{np.round(np.median(ATE_T)/cov_t_gt, 3)}"
    "$}{$"
    f"{np.round(np.degrees(np.median(ATE_R)), 3)}"
    "^{\\circ}$} & \\stackon{$"
    f"{np.round(np.median(RPE_t)/cov_t_gt, 3)}"
    "$}{$"
    f"{np.round(np.degrees(np.median(RPE_R)), 3)}"
    "^{\\circ}$}"
)

with open(output_path + "/output.txt", "w") as f:
    f.write(output_str)
        
    