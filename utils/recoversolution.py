import numpy as np
from scipy.linalg import eigh, svd

def recover_XM(Q,R,s,Abar,lam):
    N = s.shape[0]
        
    sR = np.zeros((3*N, R.shape[1]))
    for i in range(N):
        sR[3*i:3*i+3,:] = s[i] * R[3*i:3*i+3,:]
        
    if R.shape[1] > 3:
        X = sR @ sR.T

        eig_vals, eig_vecs = eigh(X)

        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        top_3_eig_vecs = eig_vecs[:, :3]

        sR_real = top_3_eig_vecs * np.sqrt(eig_vals[:3])
        sR_real = sR_real.T
        
        if(abs(eig_vals[3]/eig_vals[2]) < 1e-3):
            print("Optimal rank is 3")
        else:
            X_new = sR_real.T @ sR_real
            suboptimality = np.sum(Q * (X_new - X)) + lam * np.sum((np.diag(X_new)-1)**2)/3 - lam * np.sum((np.diag(X) - 1)**2)/3
            print("suboptimality: ", suboptimality)

    else:
        s_real = s.squeeze()
        R_real = R.T
        sR_real = np.zeros((3, 3*N))
        for i in range(N):
            sR_real[:,3*i:3*i+3] = s_real[i] * R_real[:,3*i:3*i+3]
        
    s_real = np.zeros(N)
    R_real = np.zeros((3, 3*N))

    for i in range(N):
        s_real[i] = np.linalg.norm(sR_real[:, 3*i:3*i+3], 'fro') / np.sqrt(3)
        R_real[:, 3*i:3*i+3] = sR_real[:, 3*i:3*i+3] / s_real[i]
    
    # because of anchoring
    R1 = R_real[:,:3]
    R_real = R1.T @ R_real

    negative_R = 0
    for i in range(N):
        U, S, Vt = svd(R_real[:,3*i:3*i+3])

        if np.linalg.det(U @ Vt) < 0:
            negative_R = negative_R + 1

    # not projecting the SO3, sometimes worse than the original in Ceres
    if negative_R > 0:
        print("warning: some det(R) < 0")

    # judge which is the largest component
    if negative_R > N / 2:
        R_real = -R_real

    for i in range(N):
        U, S, Vt = svd(R_real[:,3*i:3*i+3])

        if np.linalg.det(U @ Vt) < 0:
            R_real[:,3*i:3*i+3] = U @ Vt
            sR_real[:,3*i:3*i+3] = s_real[i] * R_real[:,3*i:3*i+3]
        else:
            R_real[:,3*i:3*i+3] = U @ Vt
            sR_real[:,3*i:3*i+3] = s_real[i] * R_real[:,3*i:3*i+3]
        
    # Compute ybar_est
    ybar_est = Abar @ sR_real.T

    # Add a column of zeros to the left
    y_est = np.hstack((np.zeros((3, 1)), ybar_est.T))

    # Compute final outputs
    t_est = y_est[:, :N]
    p_est = y_est[:, N:]
    
    return R_real, s_real, p_est, t_est

