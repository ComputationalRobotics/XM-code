import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'XM/build/'))

import XM

from utils.creatematrix import create_matrix
from utils.io import save_matrix_to_bin, load_matrix_from_bin
from utils.recoversolution import recover_XM
from utils.visualization import visualize_camera, visualize
from utils.checkconnection import checklandmarks
from utils.readgt_BAL import load_BAL_gt
from utils.error import ATE_TEASER_C2W

import numpy as np
import networkx as nx
import random

############################
# For experiment in paper only
# download the dataset from:
# https://drive.google.com/drive/folders/13_2mcKGKVU0ibWck2n4ajUrN2MaDfR7y?usp=sharing
# and put it in the folder './assets/Experiment/**'
############################

# # BAL datasets: BAL-93, BAL-392, BAL-1934
# dataset_path = './assets/Experiment/BAL/BAL-1934'

dataset_path = './assets/SIMPLE2'

data, n_observation = load_matrix_from_bin(dataset_path + '/landmark.bin')

# dalete the row that contains the same edges in view-graph
edges = data[:,:2].astype(int)
_, unique_indices = np.unique(edges, axis=0, return_index=True)
edges = edges[unique_indices,:]
data = data[unique_indices,:]

weights = data[:,5]
landmarks = data[:,2:5]

gt = load_BAL_gt(dataset_path)

"""
    input data:
        edges (np.ndarray): A 2D array of shape (num_ob, 2) where each row represents an edge in the view graph.
                            Each edge is defined as [frame_index, landmark_index].
                            For example:
                                edges = [[1, 1],
                                         [1, 2],
                                         [1, 3],
                                         ...]
        weights (np.ndarray): A 2D array of shape (num_ob, 1) where each row is the weight corresponding to the
                              respective edge in 'edges'.
        landmarks (np.ndarray): A 2D array of shape (num_ob, 3) where each row contains the 3D position of a landmark.
                                Note that the landmark coordinates have already been normalized using the camera intrinsics.

    Example:
        >>> edges = np.array([[1, 1],
                                [1, 2],
                                [1, 3]])
        >>> weights = np.array([[0.8],
                                [0.9],
                                [0.85]])
        >>> landmarks = np.array([[0.5, 0.1, 1.2],
                                  [0.6, 0.2, 1.1],
                                  [0.55, 0.15, 1.3]])
"""


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
max_frame, N, indices_frame = delete_thereshold(0, N, edges[:,0]-1)

# exchange the first frame (optional)
if indices_frame[max_frame] != 0:
    indices_frame[indices_frame == 0] = indices_frame[max_frame]
    indices_frame[max_frame] = 0
    
edges[:,0] = indices_frame[edges[:,0]-1].copy() + 1
# delete the row that edges contain -1
indices = np.any(edges == 0, axis=1)
edges = edges[~indices]
weights = weights[~indices]
landmarks = landmarks[~indices]

# record the index in origin input data
indices_all = indices_frame.copy()
N_old = np.where(indices_all > -1)[0].shape[0]
indices_all_copy = indices_all.copy()
for i in range(N_old):
    indices_all[np.where(indices_all_copy == i)[0]] = indices_frame[i]

# delete and reindex the landmarks that contains one frame
_, M, indices_landmarks = delete_thereshold(1, M, edges[:,1]-1)
edges[:,1] = indices_landmarks[edges[:,1]-1].copy() + 1
# delete the row that edges contain -1
indices = np.any(edges == 0, axis=1)
edges = edges[~indices]
weights = weights[~indices]
landmarks = landmarks[~indices]

# check if the graph is connected
G = nx.Graph()
for u, v in edges:
    G.add_edge(u, v + N)
components = list(nx.connected_components(G))
print("Number of connected components: ", len(components))
largest_component = max(components, key=len)
largest_component_set = set(largest_component)
filtered_indices = [
    i for i, (u, v) in enumerate(edges)
    if u in largest_component_set and (v + N) in largest_component_set
]
filtered_indices = np.array(filtered_indices)
if filtered_indices.shape[0] < edges.shape[0]:
    print("Not connected, Choose Largest Component")
    edges = edges[filtered_indices]
    weights = weights[filtered_indices]
    landmarks = landmarks[filtered_indices]
    _, N, indices_frame = delete_thereshold(0, N, edges[:,0]-1)
    edges[:,0] = indices_frame[edges[:,0]-1].copy() + 1

    N_old = np.where(indices_all > -1)[0].shape[0]
    indices_all_copy = indices_all.copy()
    for i in range(N_old):
        indices_all[np.where(indices_all_copy == i)[0]] = indices_frame[i]

    _, M, indices_landmarks = delete_thereshold(0, M, edges[:,1]-1)
    edges[:,1] = indices_landmarks[edges[:,1]-1].copy() + 1
    # here do not need to delete edges because we already know the graph is connected

# interface to create the matrix given the edges, weights and landmarks
create_matrix(weights, edges, landmarks, dataset_path)
lam = 0.0
XM.solve(dataset_path, 5, 1e-1, lam, 1000)

# visualize the camera poses
Abar,_ = load_matrix_from_bin(dataset_path + '/Abar.bin')
R,_ = load_matrix_from_bin(dataset_path + '/R.bin')
s,_ = load_matrix_from_bin(dataset_path + '/s.bin')
Q,_ = load_matrix_from_bin(dataset_path + '/Q.bin')        

# recover p and t
# details refer to our paper
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

# calculate accuracy
data_RPE_R = []
data_RPE_T = []
data_ATE_R = []
data_ATE_T = []
t_gt = np.zeros((3, N))
R_gt = np.zeros((3, 3 * N))
for i in range(N):
    i_index = np.where(indices_all == i)[0][0]
    t_gt[:,i] = gt[i_index]["t"]  
    R_gt[:,3*i:3*i+3] = gt[i_index]["R"]

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
