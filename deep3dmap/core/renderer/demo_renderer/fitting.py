import numpy as np
import matplotlib.pyplot as plt

from mesh.vertices import *
from geometry.camera import *
from time import time

def fit_points(x, X_ind, model, max_iter = 4):
    '''
    Args:
        x: (2, n) image points
        X_ind: (n,) corresponding Model vertices index
        model: 3DMM
        max_iter: iteration
    Returns:
        pp: (6, 1). pose parameters
        sp: (199, 1). shape parameters
        ep: (29, 1). exp parameters
    '''

    #-- init
    sp = np.zeros((199, 1), dtype = np.float32)
    ep = np.zeros((29, 1), dtype = np.float32)

    #-------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')
    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :]
    expPC = model['expPC'][valid_ind, :]

    stf = time()
    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [len(X)/3, 3]).T
        
        # if i>0:
        #     x_hat = scaled_orthog_project(X, s, R, t2d)
        #     plt.plot(x[0,:], 128-1-x[1,:], 'yx')
        #     plt.plot(x_hat[0,:], 128-1-x_hat[1,:], 'b.')
        # plt.show()
        #----- estimate pose
        P = estimate_affine_matrix(x, X)
        s, R, t2d = P2sRt(P)
        #print 'Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t2d[0], t2d[1])

        #----- estimate shape
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [len(shape)/3, 3]).T
        ep = estimate_expression(x, shapeMU, expPC, model['expEV'], shape, s, R, t2d, lamb = 10)

        # shape
        expression = expPC.dot(ep)
        expression = np.reshape(expression, [len(expression)/3, 3]).T
        sp = estimate_shape(x, shapeMU, shapePC, model['shapeEV'], expression, s, R, t2d, lamb = 10)

    rx, ry, rz = matrix2angle(R)
    pose_para = np.array([s, rx, ry, rz, t2d[0], t2d[1]])[:, np.newaxis]

    return pose_para, sp, ep


def fit_points_simple(x, X_ind, model, max_iter = 4):
    '''
    Args:
        x: (2, n) image points
        X_ind: (n,) corresponding Model vertices index
        model: 3DMM
        max_iter: iteration
    Returns:
        pp: (6, 1). pose parameters
        sp: (199, 1). shape parameters
        ep: (29, 1). exp parameters
    '''
    n_sp = 100
    n_ep = 25
    #-- init
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep = np.zeros((n_ep, 1), dtype = np.float32)

    #-------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')


    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    stf = time()
    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [len(X)/3, 3]).T
        
        # if i>0:
        #     x_hat = scaled_orthog_project(X, s, R, t2d)
        #     plt.plot(x[0,:], 128-1-x[1,:], 'yx')
        #     plt.plot(x_hat[0,:], 128-1-x_hat[1,:], 'b.')
        # plt.show()
        #----- estimate pose
        P = estimate_affine_matrix(x, X)
        s, R, t2d = P2sRt(P)
        #print 'Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t2d[0], t2d[1])

        #----- estimate shape
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [len(shape)/3, 3]).T
        ep = estimate_expression(x, shapeMU, expPC, model['expEV'][:n_ep,:], shape, s, R, t2d, lamb = 20)

        # shape
        expression = expPC.dot(ep)
        expression = np.reshape(expression, [len(expression)/3, 3]).T
        sp = estimate_shape(x, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t2d, lamb = 40)

    rx, ry, rz = matrix2angle(R)
    pose_para = np.array([s, rx, ry, rz, t2d[0], t2d[1]])[:, np.newaxis]

    sp_f = np.zeros((199, 1), dtype = np.float32); sp_f[:n_sp,:] = sp;
    ep_f = np.zeros((29, 1), dtype = np.float32);ep_f[:n_ep,:] = ep;
    return pose_para, sp_f, ep_f