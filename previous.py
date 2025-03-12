import pycolmap
import os
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import pandas as pd
from scipy.sparse import coo_matrix, save_npz, load_npz
from scipy.spatial.transform import Rotation as scipyR
from scipy.linalg import eigh, svd
import readgt_colmap
import readgt_replica
import readgt_TUM
import readgt_c3vd
from utils.plot_camera import plot_camera_axes
from joblib import Parallel, delayed
import random
from itertools import combinations
import re
import concurrent.futures
import networkx as nx

from tqdm import tqdm
import open3d as o3d
import pickle
from scipy.stats import trim_mean
import itertools

from UniDepth.unidepth.models import UniDepthV1, UniDepthV2
import torch

import sys
import contextlib
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./Depth-Anything-V2/metric_depth/"))
import XM

from utils.io import load_c_matrix_from_bin, save_c_matrix_to_bin, save_array_to_bin, Tee, load_int_matrix_from_bin
from utils.render_camera import save_camera_rotation_video, save_camera_slam_video
from utils.error import ATE_TEASER_C2W, ATE_LEASTSQUARE, rad_error
from utils.convertCOLMAP import write_cameras_text, write_images_text, write_points3D_text

from creatematrix import create_matrix
from Ceres import XM_Ceres

import teaserpp_python
from scipy.stats import chi2

import depth_pro

from depth_anything_v2.dpt import DepthAnythingV2


all_start = time.time()

dataset_name = "MipNerf/garden"
# IMCtourism, MIPnerf, replica, TUM, c3vd, c3vd-gt
dataset_category = "MIPnerf"

# unidepth, depthpro, depthanythingv2, metric3dv2
depth_model = "unidepth"


image_dir = "assets/" + dataset_name + "/images"
dataset_path = "assets/" + dataset_name
output_path = "assets/" + dataset_name + "/output"
output_path_sub = "assets/" + dataset_name + "/output/1"
database_path = output_path + "/database.db"
if dataset_category == "IMCtourism" or dataset_category == "MIPnerf":
    gt_path = "assets/" + dataset_name + "/sparse"

if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(output_path_sub):
    os.makedirs(output_path_sub)

# filter outliers directly using glomap pose
glomap_filter_start = time.time()
if is_filter_glomap:
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
        # estimate scale
        dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
        src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
        dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
        src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
        # delete 10% outliers
        index = src_dis < np.percentile(src_dis, 90)
        src_n = src[:, index]
        dst_n = dst[:, index]
        dst_avg = np.mean(dst_n, axis=1)
        src_avg = np.mean(src_n, axis=1)
        
        scale1 = np.mean(np.linalg.norm(dst_n - dst_avg.reshape(3, 1), axis=0))
        scale2 = np.mean(np.linalg.norm(src_n - src_avg.reshape(3, 1), axis=0))

        src = src / scale2 * scale1
        # because we have scale, so this may differs
        
        src_noR = R @ src
        translation = np.mean(dst - src_noR, axis=1)
        
        target = src_noR + translation.reshape(3, 1)

        # relative error
        error = np.linalg.norm(target - dst, axis=0)/scale1
        error_angle = np.arccos(np.sum(target * dst, axis=0) / (np.linalg.norm(target, axis=0) * np.linalg.norm(dst, axis=0)))
        
        percentage = np.sum(error < 0.05)/error.shape[0]

        # if percentage > 0.5:
        #     ratio = 20
        # elif percentage > 0.2:
        #     ratio = 30
        # else:
        #     ratio = 100

        if dataset_category == "IMCtourism" or dataset_category == "MIPnerf" or dataset_category == "TUM":
            ratio = 3
        else:
            ratio = 100

        thereshold = ratio * np.median(error)
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

                # 创建连接点的索引
                # 连接 src_o3d 中的点与 dst_o3d 中的点
                num_points = target.shape[1]
                line_indices = [[i, i + num_points] for i in range(num_points)]
                lines.lines = o3d.utility.Vector2iVector(line_indices)

                # 设定线段颜色
                lines.paint_uniform_color([0, 1, 0])  # 绿色
                
                src_o3d = o3d.geometry.PointCloud()
                src_o3d.points = o3d.utility.Vector3dVector(target_inlier.T)
                src_o3d.paint_uniform_color([0, 0, 1])  # 蓝色

                dst_o3d = o3d.geometry.PointCloud()
                dst_o3d.points = o3d.utility.Vector3dVector(dst_inlier.T)
                dst_o3d.paint_uniform_color([1, 0, 0])  
                
                src_O_o3d = o3d.geometry.PointCloud()
                src_O_o3d.points = o3d.utility.Vector3dVector(target_outlier.T)
                src_O_o3d.paint_uniform_color([0, 0, 0])  # 蓝色

                dst_O_o3d = o3d.geometry.PointCloud()
                dst_O_o3d.points = o3d.utility.Vector3dVector(dst_outlier.T)
                dst_O_o3d.paint_uniform_color([0, 0, 0]) 

                # Visualize:
                o3d.visualization.draw_geometries([src_o3d, dst_o3d ,lines, src_O_o3d, dst_O_o3d])
    for i in range(M):
        involved_index = np.where(error_sum[:,i]>0)[0]
        if len(involved_index) > 0:
            max_involved = np.max(error_sum[involved_index,i])
            outliers = np.where(error_sum[involved_index,i] > 0.25 * max_involved)[0]
            is_outlier[vis[involved_index[outliers], i].toarray().squeeze()] = True
            # print("landmark ", i, "error sum: ", error_sum[np.where(error_sum[:,i]>0),i], "total frames", vis.getcol(i).nnz)
    print("GLOMAP filter time: ", time.time() - glomap_filter_start)
    info.glomap_filter_time = time.time() - glomap_filter_start    
    edges = edges[~is_outlier]
    weights = weights[~is_outlier]
    landmarks = landmarks[~is_outlier]
    rgbs = rgbs[~is_outlier]



    print("Total remain observations after glomap pose:", edges.shape[0])
    print("Total delete observations after glomap pose:", np.sum(is_outlier))
    # save data
    np.save(output_path_sub + "/weights_glomap.npy", weights)
    np.save(output_path_sub + "/edges_glomap.npy", edges)
    np.save(output_path_sub + "/landmarks_glomap.npy", landmarks)
    np.save(output_path_sub + "/rgbs_glomap.npy", rgbs) 
else:
    # judge if they exist
    if os.path.exists(output_path_sub + "/weights_glomap.npy"):
        weights = np.load(output_path_sub + "/weights_glomap.npy")
        edges = np.load(output_path_sub + "/edges_glomap.npy")
        landmarks = np.load(output_path_sub + "/landmarks_glomap.npy")
        rgbs = np.load(output_path_sub + "/rgbs_glomap.npy")
        N = edges[:,0].max()
        M = edges[:,1].max()


    inliers = []
    if is_ceres:
        landmarks_2D = landmarks[:, :2] / landmarks[:, 2, None]
        nan_mask = np.isnan(landmarks_2D).any(axis=1)

        R_real, t_est, p_est, ceres_time = XM_Ceres(edges[~nan_mask], landmarks_2D[~nan_mask], R_real, t_est, p_est)
        
        sR_real = np.zeros((3, 3*N))
        for i in range(N):
            sR_real[:,3*i:3*i+3] = s_real[i] * R_real[:,3*i:3*i+3]
            
        # Compute ybar_est
        ybar_est = Abar @ sR_real.T
        
        # Add a column of zeros to the left
        y_est = np.hstack((np.zeros((3, 1)), ybar_est.T))

        p_est = y_est[:, N:N + M]
        t_est_visualization = y_est[:, :N]
        

        # R_real, t_est, p_est, ceres_time = XM_Ceres(edges[~nan_mask], landmarks_2D[~nan_mask], R_real, t_est, p_est, only_landmarks=True)
        # sR_real = np.zeros((3, 3*N))
        # for i in range(N):
        #     sR_real[:,3*i:3*i+3] = s_real[i] * R_real[:,3*i:3*i+3]
            
        # # Compute ybar_est
        # ybar_est = Abar @ sR_real.T
        
        # # Add a column of zeros to the left
        # y_est = np.hstack((np.zeros((3, 1)), ybar_est.T))

        # p_est = y_est[:, N:N + M]
            
        # R_real, t_est, p_est, ceres_time = XM_Ceres(edges[~nan_mask], landmarks_2D[~nan_mask], R_real, t_est, p_est, only_landmarks=True)
        
        # t_est_visualization = t_est.copy()
        
        # t_est_visualization = t_est.copy()
        
        # After solving we need re-estimate the scale for dense point cloud
        s_update = np.zeros(N)
        
        vis = coo_matrix(((np.arange(edges.shape[0])), (edges[:,0]-1, edges[:,1]-1)), shape=(N, M)).tocsr()
        for i in range(N):
            observation_index = vis.getrow(i).toarray().ravel()
            landmark_index = np.nonzero(observation_index)[0]
            observation_index = observation_index[landmark_index]
            
            if len(landmark_index) < 20:
                continue
            
            src = landmarks[observation_index,:].T
            dst = p_est[:,landmark_index]
            src = np.squeeze(src).copy()  # Remove unnecessary dimensions
            dst = np.squeeze(dst).copy()
            
            dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
            src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
            dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
            src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
            # delete 10% outliers
            index = src_dis < np.percentile(src_dis, 90)
            src_s = src[:, index]
            dst_s = dst[:, index]
            dst_s_avg = np.mean(dst_s, axis=1)
            src_s_avg = np.mean(src_s, axis=1)
            
            scale1 = np.mean(np.linalg.norm(dst_s - dst_s_avg.reshape(3, 1), axis=0))
            scale2 = np.mean(np.linalg.norm(src_s - src_s_avg.reshape(3, 1), axis=0))
            scale = scale1 / scale2
            
            s_update[i] = scale
            
            # test if they are good
            target = scale * R_real[:,3*i:3*i+3] @ src + t_est_visualization[:,i].reshape(3,1)
            
            #  # visualization
            # src_o3d = o3d.geometry.PointCloud()
            # src_o3d.points = o3d.utility.Vector3dVector(target.T)
            # src_o3d.paint_uniform_color([0, 0, 1])  # 蓝色

            # dst_o3d = o3d.geometry.PointCloud()
            # dst_o3d.points = o3d.utility.Vector3dVector(dst.T)
            # dst_o3d.paint_uniform_color([1, 0, 0]) 

            # o3d.visualization.draw_geometries([src_o3d, dst_o3d]) 
            
            error = np.linalg.norm(target - dst, axis=0)/scale1
            # print(error)
            percentage = np.sum(error < 0.2)/error.shape[0]
            if percentage > 0.8:
                print("image ", i, " is good and used for dense point cloud")
                inliers.append(i)
                
                
        
        info.ceres_time = ceres_time
        save_c_matrix_to_bin(output_path_sub + "/R_real.bin", R_real)
        save_c_matrix_to_bin(output_path_sub + "/t.bin", t_est)
        save_c_matrix_to_bin(output_path_sub + "/p.bin", p_est)
        save_c_matrix_to_bin(output_path_sub + "/s_update.bin", s_update[:,np.newaxis])
        save_c_matrix_to_bin(output_path_sub + "/t_est_vis.bin", t_est_visualization)
        np.save(output_path_sub + "/inliers.npy", inliers)
    else:
        
        t_est_visualization = t_est.copy()
        # After solving we need re-estimate the scale for dense point cloud
        s_update = np.zeros(N)
        inliers = []
        vis = coo_matrix(((np.arange(edges.shape[0])), (edges[:,0]-1, edges[:,1]-1)), shape=(N, M)).tocsr()
        for i in range(N):
            observation_index = vis.getrow(i).toarray().ravel()
            landmark_index = np.nonzero(observation_index)[0]
            observation_index = observation_index[landmark_index]
            
            src = landmarks[observation_index,:].T
            dst = p_est[:,landmark_index]
            src = np.squeeze(src).copy()  # Remove unnecessary dimensions
            dst = np.squeeze(dst).copy()
            
            dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
            src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
            dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
            src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
            # delete 10% outliers
            index = src_dis < np.percentile(src_dis, 90)
            src_s = src[:, index]
            dst_s = dst[:, index]
            dst_s_avg = np.mean(dst_s, axis=1)
            src_s_avg = np.mean(src_s, axis=1)
            
            scale1 = np.mean(np.linalg.norm(dst_s - dst_s_avg.reshape(3, 1), axis=0))
            scale2 = np.mean(np.linalg.norm(src_s - src_s_avg.reshape(3, 1), axis=0))
            scale = scale1 / scale2
            
            s_update[i] = scale
            
            # test if they are good
            target = scale * R_real[:,3*i:3*i+3] @ src + t_est_visualization[:,i].reshape(3,1)
            
            #  # visualization
            # src_o3d = o3d.geometry.PointCloud()
            # src_o3d.points = o3d.utility.Vector3dVector(target.T)
            # src_o3d.paint_uniform_color([0, 0, 1])  # 蓝色

            # dst_o3d = o3d.geometry.PointCloud()
            # dst_o3d.points = o3d.utility.Vector3dVector(dst.T)
            # dst_o3d.paint_uniform_color([1, 0, 0]) 

            # o3d.visualization.draw_geometries([src_o3d, dst_o3d]) 
            
            error = np.linalg.norm(target - dst, axis=0)/scale1
            # print(error)
            percentage = np.sum(error < 0.1)/error.shape[0]
            if percentage > 0.8:
                print("image ", i, " is good and used for dense point cloud")
                inliers.append(i)
        save_c_matrix_to_bin(output_path_sub + "/t_est_vis.bin", t_est_visualization)
        save_c_matrix_to_bin(output_path_sub + "/s_update.bin", s_update[:,np.newaxis])
        np.save(output_path_sub + "/inliers.npy", inliers)
        
    
    # # save 2D points
    # landmarks_2D = landmarks[:, :2] / landmarks[:, 2, None]
    # save_c_matrix_to_bin(
    #     output_path_sub + "/landmarks.bin",
    #     np.hstack((
    #         edges,
    #         landmarks,
    #         weights[:, np.newaxis],  # 转为 2D 数组
    #         landmarks_2D
    #     ))
    # )
else:
    t_est,_ = load_c_matrix_from_bin(output_path_sub + "/t.bin")
    p_est,_ = load_c_matrix_from_bin(output_path_sub + "/p.bin")
    R_real,_ = load_c_matrix_from_bin(output_path_sub + "/R_real.bin")
    s,_ = load_c_matrix_from_bin(output_path_sub + "/s_real.bin")
    s_update,_ = load_c_matrix_from_bin(output_path_sub + "/s_update.bin")
    mean_rgbs,_ = load_c_matrix_from_bin(output_path_sub + "/rgb.bin")
    t_est_visualization,_ = load_c_matrix_from_bin(output_path_sub + "/t_est_vis.bin")
    indices_all = np.load(output_path_sub + "/indices.npy")
    inliers = np.load(output_path_sub + "/inliers.npy")

    N = t_est.shape[1]
    M = p_est.shape[1]
    
    

# calculating the error RPE and ATE
# followed by UW-Ceres code

t_gt = np.zeros((3, N))
R_gt = np.zeros((3, 3 * N))
for i in range(N):
    i_index = np.where(indices_all == i)[0][0]
    R = gt[filename[i_index]]["R"]  # 旋转矩阵 (3x3)
    t = gt[filename[i_index]]["t"]  # 平移向量 (3x1)
    t_gt[:,i] = t
    R_gt[:,3*i:3*i+3] = R
    
save_c_matrix_to_bin(output_path_sub + "/gt_t.bin", t_gt)
save_c_matrix_to_bin(output_path_sub + "/gt_R.bin", R_gt)


# save as COLMAP format

avg_t_gt = np.mean(t_gt,axis=1)
cov_t_gt = np.mean(np.linalg.norm(t_gt - avg_t_gt.reshape(3,1),axis=0))

s_g, R_g, t_g = ATE_TEASER_C2W(R_real,t_est,R_gt,t_gt)

if dataset_category == "replica":
    cameras = readgt_replica.load_replica_camera(dataset_path)
elif dataset_category == "IMCtourism" or dataset_category == "MIPnerf":
    cameras = readgt_colmap.load_colmap_camera(gt_path)
elif dataset_category == "TUM":
    cameras = readgt_TUM.load_TUM_camera(dataset_path)
elif dataset_category == "c3vd" or dataset_category == "c3vd_gt":
    cameras = readgt_c3vd.load_c3vd_camera(dataset_path)
    
write_cameras_text(cameras, output_path_sub + "/cameras.txt")
images = {}
for i in range(N):
    i_index = np.where(indices_all == i)[0][0]
    R = R_real[:,3*i:3*i+3].T
    t = - R @ t_est[:,i]
    # R = R_gt[:,3*i:3*i+3] @ R_g
    # t = (t_gt[:,i] + R_gt[:,3*i:3*i+3] @ t_g.flatten())/s_g
    # R_real[:,3*i:3*i+3] = R.T
    # t_est[:,i] = -R.T @ t
    q = scipyR.from_matrix(R).as_quat(scalar_first=True)
    images[filename[i_index]] = {
        "qvec": q,
        "tvec": t,
        "camera_id": gt[filename[i_index]]["camera_id"],
        "name": filename[i_index],
        "id": i + 1
    }
write_images_text(images, output_path_sub + "/images.txt")
points3D = {}
for i in range(M):
    points3D[i] = {
        "xyz": p_est[:,i],
        "rgb": mean_rgbs[i,:].astype(np.uint8),
        "error": 0,
        "id": i + 1
    }
    

if use_dense:
    if os.path.exists(output_path_sub + "/p_est_dense.bin"):
        p_est,_ = load_c_matrix_from_bin(output_path_sub + "/p_est_dense.bin")
        mean_rgbs,_ = load_c_matrix_from_bin(output_path_sub + "/mean_rgbs_dense.bin")
    else:
        # adding points from depth map
        points_index = M
        print("adding points from depth map")
        p_new = []
        rgb_new = []
        if dataset_category == "IMCtourism" or dataset_category == "MIPnerf" or dataset_category == "TUM":
            print("Torch version:", torch.__version__)
            name = "unidepth-v2-vitl14"
            model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.resolution_level = 10
            
            
        for i in tqdm(range(N), desc="Processing images depth"):
            if i not in inliers:
                continue
            i_index = np.where(indices_all == i)[0][0]
            if dataset_category == "IMCtourism" or dataset_category == "MIPnerf" or dataset_category == "TUM":
                rgb = cv2.imread(image_dir + f"/{filename[i_index]}")
                rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
                predictions = model.infer(rgb_torch)
                depth_pred = predictions["depth"].squeeze().cpu().numpy()
                confidence_pred = predictions["confidence"].squeeze().cpu().numpy()

            if dataset_category == "replica":
                rgb = cv2.imread(image_dir + f"/{filename[i_index]}")
                depth_pred = cv2.imread(dataset_path + '/depth/' + 'depth{:06d}.png'.format(gt[filename[i_index]]["id"]))[:,:,0].astype(np.float32)
                depth_pred = depth_pred / 10
                confidence_pred = np.ones_like(depth_pred)
                
            if dataset_category == "c3vd":
                rgb = cv2.imread(image_dir + f"/{filename[i_index]}")
                depth_pred = cv2.imread(dataset_path + '/depth_pred/' + '{:04d}.png'.format(gt[filename[i_index]]["id"]))[:,:,0].astype(np.float32)
                depth_pred = depth_pred/255.0
                confidence_pred = np.ones_like(depth_pred)
                
            if dataset_category == "c3vd_gt":
                rgb = cv2.imread(image_dir + f"/{filename[i_index]}")
                depth_pred = cv2.imread(dataset_path + '/depth/' + '{:04d}.tiff'.format(gt[filename[i_index]]["id"]))[:,:,0].astype(np.float32)
                depth_pred = depth_pred / 65536.0 * 100
                confidence_pred = np.ones_like(depth_pred)
                
            
            h_image, w_image = depth_pred.shape
            # random u,v
            margin = 10
            # 创建 SIFT 对象
            sift = cv2.SIFT_create(nfeatures=300)

            # 检测关键点并计算描述子
            keypoints, descriptors = sift.detectAndCompute(rgb, None)

            # 提取关键点的坐标
            if len(keypoints) > 0:
                u, v = zip(*[kp.pt for kp in keypoints])
            else:
                u = []
                v = []
            if dataset_category == "IMCtourism":
                if len(u) <= 0:
                    continue
            else:
                u_rand = np.random.randint(margin, w_image-margin, 1000)
                v_rand = np.random.randint(margin, h_image-margin, 1000)
                u = np.concatenate((u, u_rand))
                v = np.concatenate((v, v_rand))
            u = np.array(u).astype(int)
            v = np.array(v).astype(int)
            # u = np.random.randint(margin, w_image-margin, 1000)
            # v = np.random.randint(margin, h_image-margin, 1000)

            d = depth_pred[v,u]
            w = confidence_pred[v,u]    

            valid_depth_indices = (w > 0.5) & (d > 0) & (d < 100)
            u = u[valid_depth_indices]
            v = v[valid_depth_indices]
            d = d[valid_depth_indices]
            col_indices = np.arange(len(u))
            
            w = confidence_pred[v,u]
            
            pixel_coords = np.vstack((u, v, np.ones_like(u) * 1))
            camera_coords = np.linalg.inv(gt[filename[i_index]]["K"]) @ pixel_coords
            camera_coords = camera_coords.T * d[:, np.newaxis]
            camera_coords = s_update[i] * R_real[:,3*i:3*i+3] @ camera_coords.T + t_est_visualization[:,i].reshape(3,1)

            for j in range(len(u)):
                points3D[points_index] = {
                    "xyz": camera_coords[:,j],
                    "rgb": rgb[v[j],u[j],[2,1,0]],
                    "error": 0,
                    "id": points_index + 1
                }
                points_index += 1
                p_new.append(camera_coords[:,j])
                rgb_new.append(rgb[v[j],u[j],[2,1,0]])
        p_new = np.array(p_new).T
        p_est = np.hstack((p_est, p_new))
        mean_rgbs = np.vstack((mean_rgbs, rgb_new))
        # save new points data
        save_c_matrix_to_bin(output_path_sub + "/p_est_dense.bin", p_est)
        save_c_matrix_to_bin(output_path_sub + "/mean_rgbs_dense.bin", mean_rgbs)

write_points3D_text(points3D, output_path_sub + "/points3D.txt")

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

print('avgRPE_R:',np.mean(RPE_R),"(rad)")
print('avgRPE_t:',np.mean(RPE_t),"(m)")
print('avgRPE_t_normalized:',np.mean(RPE_t)/cov_t_gt)

print('avgATE_R:',np.mean(ATE_R),"(rad)")
print('avgATE_T:',np.mean(ATE_T),"(m)")
print('avgATE_T_normalized:',np.mean(ATE_T)/cov_t_gt)




# visualization
# show p_est in open 3d
p_avg = np.mean(p_est, axis=1)
p_cov = np.linalg.norm(p_est - p_avg.reshape(3,1), axis=0)
p_cov = p_cov / np.median(p_cov)
valid_columns = abs(p_cov) <= 20
print("valid columns: ", valid_columns)
p_est = s_g * R_g @ p_est + t_g.reshape(3,1)

p_est = p_est[:, valid_columns]
mean_rgbs = mean_rgbs[valid_columns]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p_est.T)

# pcd_l = o3d.geometry.PointCloud()
# pcd_l.points = o3d.utility.Vector3dVector(p_new.T)


# 设置统一的 RGB 颜色，例如红色 (255, 0, 0)
# rgb_color = [165,28,48]  # 输入的 RGB 值范围为 [0, 255]
# rgb_normalized = [c / 255.0 for c in rgb_color]  # 归一化到 [0, 1]

# # 为每个点设置同样的颜色
# colors = np.tile(rgb_normalized, (p_est.shape[1], 1))  # 复制颜色到所有点 (Nx3)
# pcd.colors = o3d.utility.Vector3dVector(colors)

pcd.colors = o3d.utility.Vector3dVector(mean_rgbs/255.0)

# 定义相机内参
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(
    width=640,
    height=480,
    fx=525.0,
    fy=525.0,
    cx=320.0,
    cy=240.0
)

extrinsics = []
extrinsics_gt = []
axis_avg = []

for i in range(N):
    R = R_real[:,3*i:3*i+3]  # 旋转矩阵 (3x3)
    t = t_est[:,i]  # 平移向量 (3x1)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T @ R_g.T 
    extrinsic[:3, 3] = - (R.T @ R_g.T) @ (s_g * R_g @ t_est[:,i] + t_g.flatten()) 
    extrinsics.append(extrinsic)
    axis_avg.append(extrinsic[:3,1])

axis_avg = np.average(axis_avg, axis=0)
    
for i in range(N):
    R = R_gt[:,3*i:3*i+3]  # 旋转矩阵 (3x3)
    t = t_gt[:,i]  # 平移向量 (3x1)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    extrinsics_gt.append(extrinsic)
