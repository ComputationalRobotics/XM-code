import pandas as pd
import numpy as np

# This is a script only for Replica dataset
# If you are loading your own dataset, please write you own script
# If you are not comparing with groud truth, R and t is not neccessary, but OTHER is needed
# TODO: don't know intrinsics

def load_replica_gt(dataset_path):
    results = {}
    K = np.array([[600, 0, 599.5], [0, 600, 339.5], [0, 0, 1]])
    # load all image names in 'image' folder
    image_path = dataset_path + "/image/"
    # load gt
    data = np.loadtxt(dataset_path + '/traj.txt')
    N = data.shape[0]
    for i in range(N):
        name = 'frame{:06d}.jpg'.format(i)
        pose = data[i, :].reshape((4, 4))
        # transform to world-2-camera
        R = pose[:3, :3].T
        t = - pose[:3, :3].T @ pose[:3, 3]
        results[name] = {
            "id": i,
            "K": K,
            "R": R,
            "t": t,
            "camera_id": 1
        }
    return results

def load_replica_camera(dataset_path):
    results = {}
    results[1] = {
        "model": "PINHOLE",
        "width": 1200,
        "height": 680,
        "params": [600, 600, 599.5, 339.5]
    }
    return results

