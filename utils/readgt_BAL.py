import pandas as pd
import numpy as np
from utils.io import save_matrix_to_bin, load_matrix_from_bin

# This is a script only for Replica dataset
# If you are loading your own dataset, please write you own script
# If you are not comparing with groud truth, R and t is not neccessary, but OTHER is needed
# TODO: don't know intrinsics

def load_BAL_gt(dataset_path):
    results = {}
    # load all image names in 'image' folder
    gtR_PATH = dataset_path + "/gtR.bin"
    gtT_PATH = dataset_path + "/gtt.bin"
    # load gt
    gtR, _ = load_matrix_from_bin(gtR_PATH)
    gtT, _ = load_matrix_from_bin(gtT_PATH)
    N = gtT.shape[1]
    for i in range(N):
        # transform to world-2-camera
        R = gtR[:,3*i:3*(i+1)]
        t = gtT[:,i]
        results[i] = {
            "R": R,
            "t": t,
            "camera_id": 1
        }
    return results

def load_replica_camera(dataset_path):
    results = {}
    results[1] = {
        "model": "PINHOLE",
        "width": 2,
        "height": 2,
        "params": [1, 1, 1, 1]
    }
    return results

