import numpy as np
import teaserpp_python
from scipy.stats import chi2
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d
from scipy.stats import trim_mean

def average_rotation_from_3x3n(rotations):

    N = len(rotations)

    # find global rotation from quaternion

    quaternions = np.array([Rot.from_matrix(rot).as_quat() for rot in rotations])
    # change sign to all positive for w
    quaternions = np.array([q if q[3] > 0 else -q for q in quaternions])

    # avg_quaternion = np.mean(quaternions, axis=0)
    # avg_quaternion /= np.linalg.norm(avg_quaternion)

    # 初始化众数（从第一个四元数开始）
    mode_quaternion = quaternions[0]
    
    # # 迭代寻找众数
    # for _ in range(10):  # 设定最大迭代次数
    #     distances = np.linalg.norm(quaternions - mode_quaternion, axis=1)  # 计算每个四元数到 mode_quaternion 的欧氏距离
    #     weights = np.exp(-distances**2)  # 根据距离生成权重，高斯加权
    #     mode_quaternion = np.average(quaternions, axis=0, weights=weights)  # 加权平均
    #     mode_quaternion /= np.linalg.norm(mode_quaternion)  # 归一化为单位四元数

    R_avg = Rot.from_quat(mode_quaternion).as_matrix()

    return R_avg

def rad_error(R_est, R_gt):
    cosvalue = np.trace(R_est @ R_gt.T) - 1
    cosvalue = min(1, cosvalue)
    cosvalue = max(-1, cosvalue)
    return np.arccos(cosvalue)

def ATE_TEASER(R_est,t_est,R_gt,t_gt):
    # R_est: 3x3N
    # t_est: 3xN
    # R_gt: 3x3N
    # t_gt: 3xN
    # return: ATE
    N = int(R_est.shape[1]/3)
    assert R_est.shape == R_gt.shape
    assert t_est.shape == t_gt.shape
    assert N == t_est.shape[1]

    # find global transformation
    dof = 3
    MeasurementNoiseStd = 0.1
    epsilon_square = chi2.ppf(0.9999, dof) * (MeasurementNoiseStd ** 2)
    epsilon = np.sqrt(epsilon_square)

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 1
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    t_cam_gt = np.zeros((3, N))
    t_cam_est = np.zeros((3, N))
    for i in range(N):
        t_cam_gt[:, i] = R_gt[:, 3 * i:3 * i + 3].T @ (-t_gt[:, i])
        t_cam_est[:, i] = R_est[:, 3 * i:3 * i + 3].T @ (-t_est[:, i])
        
    src = t_cam_est
    dst = t_cam_gt

    # estimate scale
    dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
    src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
    dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
    src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
    # delete 10% outliers
    index = src_dis < np.percentile(src_dis, 90)
    src = src[:, index]
    dst = dst[:, index]
    dst_avg = np.mean(dst, axis=1)
    src_avg = np.mean(src, axis=1)
    
    scale1 = np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))
    scale2 = np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))

    src = src / np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))
    dst = dst / np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))

    # randomly choose 5000
    if src.shape[1] > 5000:
        idx = np.random.choice(src.shape[1], 5000, replace=False)
        src = src[:, idx]
        dst = dst[:, idx]

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src, dst)
    solution = solver.getSolution()

    src = t_cam_est
    dst = t_cam_gt

    src = src * scale1 / scale2
    dst = dst 

    target = solution.rotation @ src + scale1 * solution.translation.reshape(3,1)
    error = np.linalg.norm(target - dst, axis=0)
    print('TEASER error:',np.linalg.norm(error))
    
     # visualization
    src_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(target.T)
    src_o3d.paint_uniform_color([0, 0, 1])  # 蓝色

    dst_o3d = o3d.geometry.PointCloud()
    dst_o3d.points = o3d.utility.Vector3dVector(dst.T)
    dst_o3d.paint_uniform_color([1, 0, 0]) 

    o3d.visualization.draw_geometries([src_o3d, dst_o3d]) 

    return scale1 / scale2 , solution.rotation, scale1 * solution.translation.reshape(3,1)

def ATE_TEASER_C2W(R_est,t_est,R_gt,t_gt):
    # R_est: 3x3N
    # t_est: 3xN
    # R_gt: 3x3N
    # t_gt: 3xN
    # return: ATE
    N = int(R_est.shape[1]/3)
    assert R_est.shape == R_gt.shape
    assert t_est.shape == t_gt.shape
    assert N == t_est.shape[1]

    # find global transformation
    dof = 3
    MeasurementNoiseStd = 0.1
    epsilon_square = chi2.ppf(0.9999, dof) * (MeasurementNoiseStd ** 2)
    epsilon = np.sqrt(epsilon_square)

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.1
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    t_cam_gt = np.zeros((3, N))
    t_cam_est = np.zeros((3, N))
    for i in range(N):
        t_cam_gt[:, i] = R_gt[:, 3 * i:3 * i + 3].T @ (-t_gt[:, i])
        t_cam_est[:, i] = t_est[:, i]

    src = t_cam_est
    dst = t_cam_gt

    # estimate scale
    dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
    src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
    dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
    src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
    # delete 10% outliers
    index = (src_dis < np.percentile(src_dis, 90)) & (dst_dis < np.percentile(dst_dis, 90))
    src = src[:, index]
    dst = dst[:, index]
    dst_avg = np.mean(dst, axis=1)
    src_avg = np.mean(src, axis=1)
    
    scale1 = np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))
    scale2 = np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))

    src = src / scale2
    dst = dst / scale1
    # randomly choose 5000
    if src.shape[1] > 5000:
        idx = np.random.choice(src.shape[1], 5000, replace=False)
        src = src[:, idx]
        dst = dst[:, idx]

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src, dst)
    solution = solver.getSolution()

    src = t_cam_est
    dst = t_cam_gt

    src = src * scale1 / scale2
    dst = dst 

    target = solution.rotation @ src + scale1 * solution.translation.reshape(3,1)
    error = np.linalg.norm(target - dst, axis=0)
    print('TEASER error:',np.linalg.norm(error))
    
    # # visualization
    # src_o3d = o3d.geometry.PointCloud()
    # src_o3d.points = o3d.utility.Vector3dVector(target.T)
    # src_o3d.paint_uniform_color([0, 0, 1])  # 蓝色

    # dst_o3d = o3d.geometry.PointCloud()
    # dst_o3d.points = o3d.utility.Vector3dVector(dst.T)
    # dst_o3d.paint_uniform_color([1, 0, 0]) 

    # o3d.visualization.draw_geometries([src_o3d, dst_o3d]) 

    return scale1 / scale2 , solution.rotation, scale1 * solution.translation.reshape(3,1)


def ATE_LEASTSQUARE(R_est,t_est,R_gt,t_gt):
    # R_est: 3x3N
    # t_est: 3xN
    # R_gt: 3x3N
    # t_gt: 3xN
    # return: ATE
    N = int(R_est.shape[1]/3)
    assert R_est.shape == R_gt.shape
    assert t_est.shape == t_gt.shape
    assert N == t_est.shape[1]

    # find global transformation
    # find transformation
    rotations_target = [R_gt[:, 3 * i:3 * (i + 1)] @ R_est[:, 3 * i:3 * (i + 1)].T for i in range(N)]
    R = average_rotation_from_3x3n(rotations_target)
    rotation_error = [np.linalg.norm(R @ R_est[:, 3 * i:3 * (i + 1)] - R_gt[:, 3 * i:3 * (i + 1)]) for i in range(N)]
    print('rotation error:',np.mean(rotation_error))

    target = R @ t_est 
    target_avg = np.mean(target, axis=1)
    target = target - target_avg.reshape(3, 1)


    # find scale
    t_gt_avg = np.mean(t_gt, axis=1)
    cov_t_gt = np.mean(np.linalg.norm(t_gt - t_gt_avg.reshape(3, 1), axis=0))
    cov_t_est = np.mean(np.linalg.norm(target, axis=0))
    s = cov_t_gt / cov_t_est

    target = s * target

    # find translation
    t = t_gt - target
    t_avg = np.mean(t, axis=1)

    # (R * t_est - target_avg) * s + t_avg
    return s, R, t_avg.reshape(3, 1) - target_avg.reshape(3 , 1) * s

if __name__ == "__main__":
    N = 5
    