import torch
import numpy as np

def param2points_bfm(shape_param, exp_param, other_param, preds):
    # paras = torch.mul(preds[:228, :], label_std[:199+29, :])
    alpha = preds[:,:199].view(-1,199,1) * shape_param['sigma'].view(1,199,1)
    beta = preds[:,199:228].view(-1,29, 1) * 1.0/(1000.0 * other_param['sigma_exp'].view(1,29, 1))
    w_h=shape_param['w'].shape[0]
    w_w=shape_param['w'].shape[1]
    w_exp_h=exp_param['w_exp'].shape[0]
    w_exp_w=exp_param['w_exp'].shape[1]
    mu_shape_h=shape_param['mu_shape'].shape[0]
    mu_shape_w=shape_param['mu_shape'].shape[1]
    face_shape = torch.matmul(shape_param['w'].view(1,w_h,w_w), alpha) + torch.matmul(exp_param['w_exp'].view(1,w_exp_h,w_exp_w),
                 beta) + shape_param['mu_shape'].view(1, mu_shape_h,mu_shape_w)
    face_shape = face_shape.reshape(-1, 53215, 3)

    #obj['v'] = face_shape
    #obj['tri'] = model_shape['tri'].transpose()
    return [face_shape, preds[:,228:235]]