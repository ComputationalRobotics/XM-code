import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/'))

import XM

from utils.creatematrix import create_matrix
from utils.io import save_matrix_to_bin, load_matrix_from_bin
from utils.recoversolution import recover_XM
from utils.visualization import visualize_camera, visualize

import numpy as np

data, n_observation = load_matrix_from_bin('./assets/3-CreateMatrix/data.bin')

# dalete the row that contains the same edges in view-graph
edges = data[:,:2].astype(int)
_, unique_indices = np.unique(edges, axis=0, return_index=True)
edges = edges[unique_indices,:]
data = data[unique_indices,:]

weights = data[:,5]
landmarks = data[:,2:5]

# input format: edges, weights, landmarks
# edges is num_ob * 2 array, each row is a edge in view graph
# weights is num_ob * 1 array, each row is the weight of the edge
# landmarks is num_ob * 3 array, each row is the 3D position of the landmark
# The landmarks has already normalized by camera intrinsics

# edges = [1,1;
#          1,2;
#          1,3;
#         frame_index,landmark_index;
#         ...;]


N = int(np.max(edges[:, 0]))  
M = int(np.max(edges[:, 1]))  

def delete_thereshold(min_threshold, M, data):
    valid_frames = np.bincount(data, minlength=M)
    valid_indices = valid_frames > min_threshold
    wrong_indices = valid_frames <= min_threshold
    num_valid_frames = np.sum(valid_indices)
    frame_index = np.zeros(M, dtype=int)
    frame_index[valid_indices] = np.arange(num_valid_frames)
    frame_index[wrong_indices] = -1
    max_frame = np.argmax(valid_frames)
    return max_frame, num_valid_frames, frame_index

# delete and reindex the frames that contains zero landmarks
_, N, indices_frame = delete_thereshold(0, N, edges[:,0]-1)
edges[:,0] = indices_frame[edges[:,0]-1].copy() + 1
# delete the row that edges contain -1
indices = np.any(edges == 0, axis=1)
edges = edges[~indices]
weights = weights[~indices]
landmarks = landmarks[~indices]

# delete and reindex the landmarks that contains one frame
_, M, indices_landmarks = delete_thereshold(1, M, edges[:,1]-1)
edges[:,1] = indices_landmarks[edges[:,1]-1].copy() + 1
# delete the row that edges contain -1
indices = np.any(edges == 0, axis=1)
edges = edges[~indices]
weights = weights[~indices]
landmarks = landmarks[~indices]

create_matrix(weights, edges, landmarks, './assets/3-CreateMatrix')
lam = 0.0
XM.solve('./assets/3-CreateMatrix/', 5, 1e-3, lam, 1000)

# visualize the camera poses
Abar,_ = load_matrix_from_bin('./assets/3-CreateMatrix/Abar.bin')
R,_ = load_matrix_from_bin('./assets/3-CreateMatrix/R.bin')
s,_ = load_matrix_from_bin('./assets/3-CreateMatrix/s.bin')
Q,_ = load_matrix_from_bin('./assets/3-CreateMatrix/Q.bin')        

# recover p and t
R_real, s_real, p_est, t_est = recover_XM(Q, R, s, Abar, lam)

extrinsics = []
for i in range(N):
    # extrinsic is 4 * 4
    # extrinsic is world 2 camera, while R_real is camera 2 world
    extrinsics.append(np.vstack((np.hstack((R_real[:,3*i:3*i+3].T, -R_real[:,3*i:3*i+3].T @ t_est[:,i].reshape([3,1]))), np.array([0, 0, 0, 1]))))

# only visualize the camera poses
visualize_camera(extrinsics)

# visualize all
visualize(extrinsics, p_est.T)