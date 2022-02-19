'''
Computation of the Camera Matrix P (Chapter 7)
Given: correspondencs X<-->x
Estimate: Camera Matrix P, x = PX

Since the most used camera in 3D Face is Affine Camera.
So in this file, mainly use these two projection:
(1) For homogeneous coords
x = PX
P = K[R|t] = [M|t]
K = [f 0 0
     0 f 0
     0 0 1]
dof = 6. 

(2) For inhomogeneous coords
x = MX + t2d 
(M: the rows are the scalings of rows of a rotation matrix)
x = sP*R*X + t2d 
P = [1 0 0 
     0 1 0]

'''

import numpy as np

def estimate_affine_matrix(x, X):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. 
        https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp

    Args:
        x: (2, n). n>=4. 2d points
        X: (3, n). corresponding 3d points
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1])
    '''

    assert(x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert(n >= 4)
    #--- 1. normalization
    # 2d points
    mean = np.mean(x, 1) # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x**2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3,3))
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean*scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1) # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3,:] - X
    average_norm = np.mean(np.sqrt(np.sum(X**2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4,4))
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean*scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n*2, 8));
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])
 
    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4))
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))

    return P_Affine

'''
Estimation - projective transform (Chapter 4)
Given: pairs (x <--> x')
Estimate: H, x' = Hx

To estimate projective transform, which has dof=8
need at least 4 point correspondences.

'''

# Alg.4.2. The normalized DLT for 2d homographies
def direct_linear_transform(xs, xt)
    ''' normalized DTL for estimating H 
    Args:
        xs: (2, n), n>=4. the 2d points to be transformed.
        xt: (2, n), n>=4. the 2d transformed points.
    Returns:
        H: (3, 3), 2D homography matrix, xt = Hxs
    ''' 
    assert(xs.shape[1] == xt.shape[1]) 
    n = xs.shape[1]

    # if not homo, change to homo by adding 1
    if xs.shape[0] ==2 :
        xs = np.vstack((xs, np.ones((1, n))))
    if xt.shape[0] ==2 :
        xt = np.vstack((xt, np.ones((1, n))))

    # 1. normalize xs
    xs_mean = np.mean(xs, 1)
    xs = xs - np.tile(xs_mean[:, np.newaxis], [1, n])
    #xs_average_norm = 

    A = np.zeros((n*2,9)) #only use the first two rows
    #A[:n, 3:6] = -np.tile(xt[2,:][:, np.newaxis], [1,3]) * xs.T
    A[:n, 3:6] = -(xt[2,:] * xs).T # boradcasting automatic
    A[:n, 6:9] = (xt[1,:] * xs).T
    A[n:, 0:3] = (xt[2,:] * xs).T
    A[n:, 6:9] = -(xt[0,:] * xs).T
    U, S, V = svd(A, full_matrix = False)
    print U, S, V
    h = V[:, -1] # 9
    H = np.reshape(h, [3,3])
    return H