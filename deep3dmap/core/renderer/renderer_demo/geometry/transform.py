'''
Definition 2.9(Page 33)
projective transform / projectivity / collineation / homography

Define here: 
x for 2d points and X for 3d points. 
And both in homogeneous coordinates.

2d transform (Chapter 2)
-----------
2d <---> 2d

Table 2.1
* projective
h(x) = Hx
H:3x3 matrix, non-singular
dof = 8

* Affine 
H = [a11 a12 tx
     a21 a22 ty
       0   0  1]
dof = 6

* Similarity
H = [sr11 sr12 tx
     sr21 sr22 ty
        0    0  1]
dof = 4

* Eucludean / isometric
H = [r11 r12 tx
     r21 r22 ty
       0   0  1]
dof  = 3


3d transform (Chapter 3)
-----------
3d <---> 3d

Table 3.2
* projective
h(x) = Hx
H:4x4 matrix, non-singular
H = [A  t
     vT v]
dof = 15

* Affine 
H = [A t
     0 1]
dof = 12

* Similarity
H = [sR t
      0 1]
dof = 7

* Eucludean / isometric
H = [R t
     0 1]
dof  = 6

'''
import numpy as np
from np.linalg import svd


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