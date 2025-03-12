import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions

def XM_Ceres_interface(edges, landmarks2D, R_XM, t_XM, p_XM, only_landmarks=False):
    N = int(np.max(edges[:, 0]))  
    M = int(np.max(edges[:, 1]))  

    cam = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=1,
            height=1,
            params=np.array([1.,0.,0.]),
            # easy camera because we have already normalized
            camera_id=0,
        )

    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()

    # initial guess
    quat = []
    t = []
    points = []
    for i in range(N):
        # xyzw 
        Ri = pycolmap.Rotation3d(R_XM[:,3*i:3*i+3].T)
        quat.append(Ri.quat)
        t.append(-R_XM[:,3*i:3*i+3].T @ t_XM[:,i])

    for i in range(M):
        points.append(p_XM[:,i])

    for i in range(N):
        index = np.where(edges[:, 0] == i + 1)[0]
        if len(index) == 0:
            continue
        p2d = landmarks2D[index,:]
        p2d_index = edges[index,1]
        for j in range(len(index)):
            p = pycolmap.Point2D(p2d[j],p2d_index[j]-1)
            cost = pycolmap.cost_functions.ReprojErrorCost(cam.model, p.xy)
            params = [
                quat[i],
                t[i],
                points[p.point3D_id],
                cam.params,
            ]
            prob.add_residual_block(cost, loss, params)
        prob.set_manifold(
            quat[i], pyceres.EigenQuaternionManifold()
        )
    prob.set_parameter_block_constant(cam.params)
    
    if only_landmarks:
        for i in range(N):
            prob.set_parameter_block_constant(quat[i])  
    print(
        prob.num_parameter_blocks(),
        prob.num_parameters(),
        prob.num_residual_blocks(),
        prob.num_residuals(),
    )
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.ITERATIVE_SCHUR
    options.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI
    options.use_nonmonotonic_steps = True
    options.minimizer_progress_to_stdout = True
    options.num_threads = 100
    options.max_num_iterations = 1000
    options.eta = 1e-1
    options.max_solver_time_in_seconds = 300

    summary = pyceres.SolverSummary()

    pyceres.solve(options, prob, summary)
    print(summary.FullReport())
    
    p_est = np.array(points).T.copy()

    R_est = np.zeros((3,3*N))
    for i in range(N):
        R_est[:,3*i:3*i+3] = pycolmap.Rotation3d(quat[i]).matrix().T

    t_est = np.zeros((3,N))
    for i in range(N):
        t_est[:,i] = -R_est[:,3*i:3*i+3] @ t[i]  
    return R_est, t_est, p_est, summary.total_time_in_seconds
