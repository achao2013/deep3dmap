'''
Camera
1. camera model. 3d vertices project to 2d
2. estimate camera matrix. 2d points and corresponding 3d --> pose parameter
'''

import numpy as np
import math
from math import cos, sin

'''
Camera Models (chapter 6. in MVGCV)
Mapping between the 3D world(object space) and a 2D image.
Given: 3d points, Camera Matrix
Return: projected 2d points.

X: 3x1 for 3d coordinates. 4x1 for homogeneous coordinates.(just add 1)
x: 2x1 for 2d coordinates. 3x1 for homogeneous coordinates.

General projective camera:
x = PX, P = [M | p4]
P: 3x4 homogeneous camera projection matrix. 
dof = 11

det(M)
!=0 ,non-sigular: finite projective camera. 
=0, sigular: camera at infinity.

http://campar.in.tum.de/twiki/pub/Chair/TeachingWs09Cv2/3D_CV2_WS_2009_Reminder_Cameras.pdf
--------------
Models

1. Finite project camera
P = [M | -MC] 
rank(M) = 3
M is non-sigular

P = K[R | -RC] = K[R | t]
K = [fx  s px
      0 fy py
      0  0  1]
s: skew parameter
K: intrinsic camera parameters. 5 dof
R, C: extrinsic camera paramererss, each 3 dof
dof = 11

* CCD camera: (when s = 0. no skew.)
P = K[R | -RC] = K[R | t]
K = [fx  0 px
      0 fy py
      0  0  1]
aspect ratio = fy/fx
dof = 10

* Basic pinhole model: (when s=0, aspect ratio=1--> fy = fx)
P = K[R | -RC] = K[R | t]
K = [f 0 px
     0 f py
     0 0  1]
dof = 9


2. Camera at infinity
P = [M|t] 
M is sigular

- Affine cameras: (!! the most important in practice)

P: the last row P^(3T) is of the form [0, 0, 0, 1]
(then points at infinity are mapped to points at infinity. 
 My understanding: infinite points:[X,Y,Z,0] --> PX = [x, y ,0] also infinite points)

Conditions(when the affine camera can approminate the true camera):
(i) The depth relief is small compared to the average depth(d0)
(ii) The distance of the point from the principal ray is small

P = [m11 m12 m13 t1
     m21 m22 m23 t2
       0   0   0  1]
  = [M23  t
       0  1]
  = [M|t]
  = K[R|t]
rank(M) = 2
M: last row zero.

K = [fx  s 0
      0 fy 0
      0  0 1]
R = [r11 r12 r13
     r21 r22 r23
       0   0   0]
t = [t1
     t2
      1]
dof = 8

* Weak perspective projection:
P = K[R|t] = [M|t]
M: last row zero. the first two rows orthogonal.
K = [fx  0 0
      0 fy 0
      0  0 1]
dof = 7

!!!!!!
* Scaled orthographic projection:
P = K[R|t] = [M|t]
M: last row zero. the first two rows orthogonal and of equal norm.
K = [f 0 0
     0 f 0
     0 0 1]
dof = 6

Or for inhomogeneous coordinates:
x = MX + t2d 
(M: the rows are the scalings of rows of a rotation matrix)
x = sP*R*X + t2d 
P = [1 0 0 
     0 1 0]

---------------
Relationship between finite projective camera and orthographic camera
P_orthog = P_proj * H
H = [1 0 0 0
     0 1 0 0 
     0 0 0 1]
  
http://web.eecs.umich.edu/~silvio/teaching/EECS442_2010/lectures/lecture10.pdf
-------------------
Decomposition of camera matrix

1. Finding the camera center
For finite cameras and cameras at infinity
PC = 0

2. Finding the camera orientation and internal parameters
For a finite camera:
P = [M | -MC] = K[R | -RC] = K[R | t]
K: internal parameters. an upper-trangular matrix
R: canmera orientation. an orthogonal matrix.
So, 
decomposing M as M = KR using RQ-decomposition

'''
## For 3D Morphable Models
## often using scaled orthographic projection(SOP)
## which has 6 parameters: s(scale, 1) R(rotation matrix, 3) t2d(2D translation, 2)

## Actually, this can be seen as 
## a similarity transform in 3d space (see table 3.2, which has 7 dof)
## then maps parallel world lines to parallel image lines.
## because we don't use the depth in 2d image, so ty can be ignored.
## then also has the same 6 paramters.

def initial_sRt(rx = 0, ry = 0, rz = 0):
  '''
  '''
  s = 6e-04
  R = angle2matrix(rx, ry, rz)
  t2d = [66, 66]

  return s, R, t2d

def initial_pp(rx = 0, ry = 0, rz = 0):
  '''
  '''
  s = 6e-04
  t2d = [66, 66]
  pp = np.array([s, rx, ry, rz, t2d[0], t2d[1]])[:, np.newaxis]

  return pp

def scaled_orthog_project(vertices, s, R, t2d, isKeepZ = False):
    ''' scaled orthographic projection
        assumes: variations in depth over the object is small relative to the mean distance from camera to object
    
    Args:
        vertices: (3,nver). 
        s: (1,). 
        R: (3,3). 
        t2d: (2,).  
        isKeepZ

    Returns:
        projected_vertices: (2,nver)
    '''
    nver = vertices.shape[1]
    t2d = np.squeeze(np.array(t2d))

    if isKeepZ:
        t3d = np.ones(3)
        t3d[:2] = t2d
        projected_vertices = s * R.dot(vertices) + np.tile(t3d[:, np.newaxis], [1, nver])
    else:
        P = np.array([[1, 0, 0], [0, 1, 0]])
        projected_vertices = s * P.dot(R.dot(vertices)) + np.tile(t2d[:, np.newaxis], [1, nver])

    return projected_vertices


def project(vertices, pp, isKeepZ = False):
    [s, rx, ry, rz, t2dx, t2dy] = pp
    R = angle2matrix(rx, ry, rz)
    t2d = [t2dx, t2dy]
    projected_vertices = scaled_orthog_project(vertices, s, R, t2d, isKeepZ)
    return projected_vertices

def project_3ddfa(vertices, pp, isKeepZ = False):
    [rx, ry, rz, t2dx, t2dy, t2dz, s] = pp
    R = angle2matrix_3ddfa(rx, ry, rz)
    t2d = [t2dx - 1, t2dy - 1]
    projected_vertices = scaled_orthog_project(vertices, s, R, t2d, isKeepZ)
    return projected_vertices

def project_3ddfa_128(vertices, pp, isKeepZ = False):
    [rx, ry, rz, t2dx, t2dy, t2dz, s] = pp
    R = angle2matrix_3ddfa(rx, ry, rz)
    t2d = [t2dx - 1, t2dy - 1 ]
    s = s
    projected_vertices = scaled_orthog_project(vertices, s, R, t2d, isKeepZ)
    return projected_vertices
# def similarity_transform(vertices, s, R, t3d):
#     ''' similarity transform    
#     Args:
#         vertices: (3,nver). 
#         s: (1,). 
#         R: (3,3). 
#         t2d: (3,).  
    
#     Returns:
#         projected_vertices: (2,nver)
#     '''
#     nver = vertices.shape[1]
#     t2d = np.array(t2d)
#     P = np.array([[1, 0, 0], [0, 1, 0]])
#     projected_vertices = s * P.dot(R.dot(vertices)) + np.tile(t2d[:, np.newaxis], [1, nver])

#     return projected_vertices

def affine_project(vertices, affine_matrix):
    ''' scaled orthographic projection for homogeneous coordinates
        also can be used affine projection (as its father, haha)

    Args:
        vertices: (3,nver). 
        affine_matrix: (3, 4)
    
    Returns:
        projected_vertices: (2,nver)
    '''
    nver = vertices.shape[1]
    vertices_homo = np.vstack((vertices, np.ones((1, nver)))) # (3,nver)->(4,nver)
    projected_vertices_homo = affine_matrix.dot(vertices_homo)
    projected_vertices = projected_vertices_homo[:2,:] # or pv = P.dot(pvh)

    return projected_vertices

def P2sRt(P):
    ''' decompositing camera matrix P
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation. 
    '''
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d
'''
--------------------- Rotation Matrix
Representation of 3D rotation
https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions?oldformat=true
http://blog.csdn.net/mulinb/article/details/51227597
http://www.chrobotics.com/library/understanding-quaternions
http://xinchenghan.github.io/2016/09/22/calRotationMatrix/
https://www.learnopencv.com/rotation-matrix-to-euler-angles/

1. Euler Angle 
angle_x, angle_y, angle_z

2. Rotation Matrix (Derection Cosine Matrix)
3x3 matrix. 
dof = 3 ==> orthogonal matrix

3. Axis-Angle / Rotation Vector
theta, x, y, z
dof =3 ==> theta = norm(x, y, z)
(often used in OpenCV)

4. Quaternion 
[cos(theta/2), x*sin(theta/2), y*sin(theta/2), z*sin(theta/2)]


####### In 3D face. 
Often use two representation: Euler Angle and Rotation Matrix.
##  Euler Angle
r_x, r_y, r_z: three angles
For easily discribe the pose of head.

## Rotation Matrix
R: 3x3 matrix
For projection/rotation calculation

!!!!!!!!!!
Define here: 
see https://www.researchgate.net/profile/Tsang_Ing_Ren/publication/279291928/figure/fig1/AS:292533185462272@1446756754388/Fig-1-Orientation-of-the-head-in-terms-of-pitch-roll-and-yaw-movements-describing.png
1. angle setting
r_x = pitch: positive for looking down 
r_y = yaw: positive for looking left
r_z = roll: positive for tilting head right

2. 3d coords and 2d image
u = x (width, shape[1])
v = height - 1 - y (height, shape[0])
depth = z (bigger for fonter)

3. the conversion between Euler Angle and Rotation Matrix
* Angle --> Matrix
Using right-hand system.
Rx = [[1,         0,          0],
      [0, np.cos(x),  -np.sin(x)],
      [0, np.sin(x),  np.cos(x)]]
Ry = [[ np.cos(y), 0, np.sin(y)],
      [         0, 1,         0],
      [-np.sin(y), 0, np.cos(y)]]
Rz = [[np.cos(z), -np.sin(z), 0],
      [np.sin(z),  np.cos(z), 0],
      [        0,          0, 1]]
Define rotation order: X first, then Y, then Z (industry standard)
Rotation Matrix = Rz * Ry * Rx

Rotated_Vertex = R * Vertex. 
The order is important, the result will change if the order changes.

* Matrix --> Angle
https://www.learnopencv.com/rotation-matrix-to-euler-angles/

'''

def angle2matrix(x, y, z):
    ''' get rotation matrix from three rotation angles
    Args:
        x: pitch. positive for looking down 
        y: yaw. positive for looking left
        z: roll. positive for tilting head right
    Returns:
        R: 3x3. rotation matrix.
    '''
    
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    return R

def angle2matrix_3ddfa(x, y, z):
    ''' get rotation matrix from three rotation angles
    Args:
        x: pitch. positive for looking down 
        y: yaw. positive for looking left
        z: roll. positive for tilting head right
    Returns:
        R: 3x3. rotation matrix.
    '''
    
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  sin(x)],
                 [0, -sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, -sin(y)],
                 [      0, 1,      0],
                 [sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), sin(z), 0],
                 [-sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rx.dot(Ry.dot(Rz))
    return R


#Ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert(isRotationMatrix)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    # if y > np.pi/2:
        
    # x = (x + np.pi) % np.pi
    # z = (z + np.pi) % np.pi
    return x, y, z


''' -------------------------------------- estimate
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

    T = np.zeros((3,3), dtype = np.float32)
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

    U = np.zeros((4,4), dtype = np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean*scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n*2, 8), dtype = np.float32);
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])
 
    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype = np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))
    return P_Affine