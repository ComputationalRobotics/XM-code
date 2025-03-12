import pyceres
import pycolmap

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'XM/build/'))
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

import XM

from scipy.sparse import coo_matrix, save_npz, load_npz

from utils.readgt_replica import load_replica_gt, load_replica_camera
from utils.cameramath import quat2rot
from utils.checkconnection import checklandmarks
from utils.creatematrix import create_matrix
from utils.io import save_matrix_to_bin, load_matrix_from_bin
from utils.recoversolution import recover_XM
from utils.visualization import visualize_camera, visualize

current_dir = os.path.dirname(os.path.abspath(__file__))
print( os.path.abspath(os.path.join(current_dir, "./assets/Replica/images")))
image_dir = os.path.abspath(os.path.join(current_dir, "./assets/Replica/images"))
dataset_path = os.path.abspath(os.path.join(current_dir, "./assets/Replica/"))
output_path = os.path.abspath(os.path.join(current_dir, "./assets/Replica/output"))
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

# load camera information and gt poses (if needed)
gt = load_replica_gt(dataset_path)
gt_camera = load_replica_camera(dataset_path)

# Run COLMAP feature extracting and matching
if run_colmap:
    if len(gt_camera) == 1:
        # single camera
        ImageReaderOptions = pycolmap.ImageReaderOptions()
        ImageReaderOptions.camera_params = '600,600,599.5,339.5'
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

    matches = np.loadtxt(os.path.abspath(os.path.join(current_dir, "../../assets/tempdata/output.txt"))) 
    File3 = pd.read_table(os.path.abspath(os.path.join(current_dir, "../../assets/tempdata/filename.txt")), header=None, names=["Var1", "Var2"], sep='\s+')
    relpose = np.loadtxt(os.path.abspath(os.path.join(current_dir, "../../assets/tempdata/relative_pose.txt")))

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
    for i in tqdm(range(N), desc="Processing images depth"):
        rgb = cv2.imread(image_dir + f"/{filename[i]}")
        depth_pred = cv2.imread(dataset_path + '/depth/' + 'depth{:06d}.png'.format(gt[filename[i]]["id"]))[:,:,0].astype(np.float32)
        depth_pred = depth_pred / 10
        confidence_pred = np.ones_like(depth_pred)
        
        col_indices = vis[i].nonzero()[1]
        if len(col_indices) == 0:
            continue
        point_frame = []
        h_image, w_image = depth_pred.shape
        u = landmarkx.getrow(i).data.astype(int)
        v = landmarky.getrow(i).data.astype(int)
        
        valid_indices = (u >= 0) & (u < w_image) & (v >= 0) & (v < h_image)
        u = u[valid_indices]
        v = v[valid_indices]
        col_indices = col_indices[valid_indices]
        if u.shape[0] == 0:
            continue    
        
        d = depth_pred[v, u] 
        
        valid_depth_indices = (d > 0)
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
        
# YOUR OWN FILTER HERE

# send it to XM
# check is the view-graph is connected
edges, landmarks, weights, rgbs = checklandmarks(edges, landmarks, weights, rgbs, N, M)

create_matrix(weights, edges, landmarks, output_path)
lam = edges.shape[0] / N
XM.solve(output_path, 5, 1e-3, lam, 1000)

# visualize the camera poses
Abar,_ = load_matrix_from_bin(output_path + '/Abar.bin')
R,_ = load_matrix_from_bin(output_path + '/R.bin')
s,_ = load_matrix_from_bin(output_path + '/s.bin')
Q,_ = load_matrix_from_bin(output_path + '/Q.bin')        

# recover p and t
R_real, s_real, p_est, t_est = recover_XM(Q, R, s, Abar, lam)

N = s_real.shape[0]
M = p_est.shape[1]

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

# visualize all
visualize(extrinsics, p_est.T, mean_rgbs/255.0)
        
    