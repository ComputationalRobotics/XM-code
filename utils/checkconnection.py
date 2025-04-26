import numpy as np
import networkx as nx

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

def checklandmarks(edges, landmarks, weights, rgbs, N, M):
    # delete and reindex the frames that contains zero landmarks
    # Note: change the thereshold higher if you do not get good result, e.g. IMC gate
    max_frame, N, indices_frame = delete_thereshold(10, N, edges[:,0]-1)

    # exchange the first frame
    if indices_frame[max_frame] != 0:
        indices_frame[indices_frame == 0] = indices_frame[max_frame]
        indices_frame[max_frame] = 0

    indices_all = indices_frame.copy()  
    edges[:,0] = indices_frame[edges[:,0]-1].copy() + 1
    # delete the row that edges contain -1
    indices = np.any(edges == 0, axis=1)
    edges = edges[~indices]
    weights = weights[~indices]
    landmarks = landmarks[~indices]
    rgbs = rgbs[~indices]

    # delete and reindex the landmarks that contains one frame
    # Note: change the thereshold higher if you do not get good result, e.g. IMC gate
    _, M, indices_landmarks = delete_thereshold(1, M, edges[:,1]-1)
    edges[:,1] = indices_landmarks[edges[:,1]-1].copy() + 1
    # delete the row that edges contain -1
    indices = np.any(edges == 0, axis=1)
    edges = edges[~indices]
    weights = weights[~indices]
    rgbs = rgbs[~indices]
    landmarks = landmarks[~indices]

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
        rgbs = rgbs[filtered_indices]
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
        
    return edges, landmarks, weights, rgbs, indices_all
