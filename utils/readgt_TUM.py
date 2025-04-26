import pandas as pd
import numpy as np
import os

from utils.io import load_matrix_from_bin,save_matrix_to_bin

# The reconstructed pose of an image is specified as the projection from world to the camera coordinate
def quat2rot(qw, qx, qy, qz):
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return R

def load_tum_gt(dataset_path):

    results = {}
    image_dir = dataset_path + "/images/"
    # get all filename
    all_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    fx = 517.3  # focal length x
    fy = 516.5  # focal length y
    cx = 318.6  # optical center x
    cy = 255.3  # optical center y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # Read ground truth poses and timestamps
    gtname = dataset_path + "/groundtruth.txt"
    data = pd.read_csv(gtname, sep=' ', header=None, names=['t', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

    timestamps = data['t'].values
    txyz = data[['tx', 'ty', 'tz']].values
    q = data[['qw', 'qx', 'qy', 'qz']].values
    
    # load all image in 'image' folder
    for i in range(len(all_files)):
        timestamp = float(all_files[i].replace('.png', ''))
        insert_position = np.searchsorted(timestamps, timestamp)
        if insert_position == 0:
            q_interpolated = q[0]
            t_interpolated = txyz[0]
        elif insert_position == len(timestamps):
            q_interpolated = q[-1]
            t_interpolated = txyz[-1]
        else:
            factor = (timestamp - timestamps[insert_position - 1]) / (timestamps[insert_position] - timestamps[insert_position - 1])
            q_interpolated = (1 - factor) * q[insert_position - 1] + factor * q[insert_position]
            t_interpolated = (1 - factor) * txyz[insert_position - 1] + factor * txyz[insert_position]
        q_interpolated = q_interpolated / np.linalg.norm(q_interpolated)
        R = quat2rot(q_interpolated[0], q_interpolated[1], q_interpolated[2], q_interpolated[3]).T
        t = -R @ t_interpolated
        results[all_files[i]] = {
            "id": i,
            "K": K,
            "R": R,
            "t": t,
            "camera_id": 1
        }
    return results

def load_tum_camera(dataset_path):
    results = {}
    results[1] = {
        "model": "PINHOLE",
        "width": 640,
        "height": 480,
        "params": [517.3,516.5,318.6,255.3]
    }
    return results