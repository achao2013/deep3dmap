'''
Generate vertices
Estimating parameters about vertices: shape para, exp para
'''

import numpy as np 
from time import time 
from numba import jit

#-------  generate
def random_shape_para(n_shape_para = 199):
    sp = np.random.rand(n_shape_para, 1)*1e04
    #C = sio.loadmat('Data/para.mat')
    #sp = C['alpha']
    return sp

def random_exp_para(n_exp_para = 29):
    # ep = np.zeros((n_exp_para, 1))
    ep = -1.5 + 3*np.random.random([n_exp_para, 1])#*1e04
    ep[6:, 0] = 0

    #C = sio.loadmat('Data/para.mat')
    #ep = C['alpha_exp']
    return ep 

def generate_vertices(model, shape_para, exp_para):
    '''
    Args:
        model: 3DMM model
        shape_para: (199, 1)
        exp_para: (29, 1) 
    Returns:
        vertices: (3, nver)
    '''
    vertices = model['shapeMU'] + model['shapePC'].dot(shape_para) + model['expPC'].dot(exp_para)
    vertices = np.reshape(vertices, [3, int(len(vertices)/3)], 'F')
    return vertices
    

# -------- estimate

'''
Given: image_points, 3D Model, Camera Matrix(s, R, t2d)
Estimate: shape parameters, expression parameters

Inference: 

    projected_vertices = s*P*R(mu + shape + exp) + t2d  --> image_points
    s*P*R*shape + s*P*R(mu + exp) + t2d --> image_poitns

    # Define:
    X = vertices
    x_hat = projected_vertices
    x = image_points
    A = s*P*R
    b = s*P*R(mu + exp) + t2d
    ==>
    x_hat = A*shape + b  (2 x n)

    A*shape (2 x n)
    shape = reshape(shapePC * sp) (3 x n)
    shapePC*sp : (3n x 1)

    * flatten:
    x_hat_flatten = A*shape + b_flatten  (2n x 1)
    A*shape (2n x 1)
    --> A*shapePC (2n x 199)  sp: 199 x 1
    
    # Define:
    pc_2d = A* reshape(shapePC)
    pc_2d_flatten = flatten(pc_2d) (2n x 199)

    =====>
    x_hat_flatten = pc_2d_flatten * sp + b_flatten ---> x_flatten (2n x 1)

    Goals:
    (ignore flatten, pc_2d-->pc)
    min E = || x_hat - x || + lambda*sum(sp/sigma)^2
          = || pc * sp + b - x || + lambda*sum(sp/sigma)^2

    Solve:
    d(E)/d(sp) = 0
    2 * pc' * (pc * sp + b - x) + 2 * lambda * sp / (sigma' * sigma) = 0

    Get:
    (pc' * pc + lambda / (sigma'* sigma)) * sp  = pc' * (x - b)

'''

def estimate_shape(x, shapeMU, shapePC, shapeEV, expression, s, R, t2d, lamb = 3000):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        shapeMU: (3n, 1)
        shapePC: (3n, 199)
        shapeEV: (199, 1)
        expression: (3, n)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient

    Returns:
        shape_para: (199, 1) shape parameters(coefficients)
    '''
    assert(shapeMU.shape[0] == shapePC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = shapePC.shape[1]

    n = x.shape[1]
    sigma = shapeEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    # --- calc pc
    pc_3d = np.resize(shapePC.T, [dof, n, 3]) # 199 x n x 3
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) ## !it still works without this line, but Adding this will let the next line run faster than without reshape, I don't know why. ##ToDo: learn more about numpy dot mul  
    pc_2d = pc_3d.dot(A.T.copy()) # 199 x n x 2
    
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 199

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # expression
    exp_3d = expression
    # 
    b = A.dot(mu_3d + exp_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para

def estimate_expression(x, shapeMU, expPC, expEV, shape, s, R, t2d, lamb = 2000):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        shapeMU: (3n, 1)
        expPC: (3n, 29)
        expEV: (29, 1)
        shape: (3, n)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient

    Returns:
        exp_para: (29, 1) shape parameters(coefficients)
    '''
    assert(shapeMU.shape[0] == expPC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = expPC.shape[1]

    n = x.shape[1]
    sigma = expEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    # --- calc pc
    pc_3d = np.resize(expPC.T, [dof, n, 3]) 
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) ## !it still works without this line, but Adding this will let the next line run about 4 times faster than without reshape, I don't know why. ##ToDo: learn more about numpy dot mul  
    pc_2d = pc_3d.dot(A.T) 
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # expression
    shape_3d = shape
    # 
    b = A.dot(mu_3d + shape_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    exp_para = np.dot(np.linalg.inv(equation_left), equation_right)
    
    return exp_para


