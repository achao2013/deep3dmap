import numpy as np
import os
from scipy.misc import imread, imsave, imresize
from glob import glob
import scipy.io as sio
from time import time
from glob import glob
import tensorflow as tf

import tdmm
import tdmm.geometry.camera as camera

import tdmm.mesh as mesh
import tdmm.mesh_cython as mesh_cython

import tdmm.fitting
# from td.geometry.camera import *

import sys
import fit_geo
from fit_geo.uv_pos_1 import PosNet
from fit_geo.utils import datas
from cv_plot import plot_kpt

# load model
model = tdmm.load_model('tdmm/Data/BFM.mat')
model_info = tdmm.load_model_info('tdmm/Data/BFM_info.mat')
mean_texture = mesh.texture.generate_texture(model)

nver, ntri = tdmm.get_model_num(model)
n_shape_para, n_exp_para = tdmm.get_para_num(model)
n_tex_para = n_shape_para
triangles = tdmm.get_triangles(model)
light_triangles = tdmm.get_triangles(model, False)
kpt_ind = tdmm.get_kpt_ind(model)
organ_ind = tdmm.get_organ_ind(model_info)


# uv data
uv_face_xyz = np.loadtxt('tdmm/Data/uv_face_xyz.txt') # 3 x nver
uv_vis_ind = np.loadtxt('tdmm/Data/uv_face_vis_ind.txt') # 3 x nver
iind = np.arange(0, len(uv_vis_ind), 3)
uv_vis_ind = uv_vis_ind[iind]
X = uv_face_xyz[:2,:]
uv_vis_ind = []
for k in range(nver):
    if abs(X[1,k] - np.round(X[1,k])) < 0.2 and abs(X[0,k] - np.round(X[0,k])) < 0.2:
        uv_vis_ind.append(k)
uv_vis_ind = np.array(uv_vis_ind)
uv_vis_ind = np.union1d(uv_vis_ind, organ_ind)

# set para
h = w = 128
c = 3

# pos net
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
uv_face = imread('tdmm/Data/uv_geo/uv_face.jpg')/255.
vis_mask = imread('tdmm/Data/uv_geo/uv_vis_mask.jpg')/255.
vis_mask_single= vis_mask[np.newaxis, :,:,np.newaxis]
uv_face_single = uv_face*vis_mask_single
net = PosNet()
model_name = 'fit_geo/Models/uv_pos_1112'
# net.sess.run(tf.global_variables_initializer())
# net.saver.restore(net.sess, model_name)
tf.train.Saver(net.G.vars).restore(net.sess, model_name)


image_folder = '/home/fengyao/GIT/2/fy/result/real/'
save_folder = '/home/fengyao/GIT/2/fy/result/real/fitted_ir/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
image_list = glob(image_folder + '0_cropped_ir.jpg')
print 'image numbers: {}'.format(len(image_list))

test_data = datas.train_uv_pos()
for i, image_name in enumerate(image_list):
    #---------- 0. read images
    image = imread(image_name)
    image = imresize(image, [128, 128])/255.
    image = image.astype(np.float32)
    image_name = image_name.replace('png', 'jpg')
    imsave(image_name.replace(image_folder, save_folder), np.squeeze(image))

    #---------- 1. use CNN get position
    stc = time()
    pos, vis = net.sess.run([net.xf_fake_test, net.xm_fake_test], feed_dict = {net.x: image[np.newaxis, :,:,:], net.uv_face: uv_face_single})
    print 'cnn time:', time() - stc

    # imsave(image_name.replace('_test/', '_test/flow_npy/').replace('.jpg', '_flow.jpg'), flow)
    # imsave(image_name.replace(image_folder, '_test/flow_npy/').replace('.jpg', '_match.jpg'), match)
    
    #--------- 2. get x, Xind pairs
    pos_in_uv = pos[0,:,:,:]*127.
    vis_in_uv = (vis[0,:,:,0] > 0.15) 
    X = uv_face_xyz[:2, :]
    vis = vis_in_uv[np.round(X[1,:]).astype(np.int32), np.round(X[0,:]).astype(np.int32)]
    X_vis_ind = np.nonzero(vis)[0]
    # mean_in_vis_ind
    uv_vis_ind = np.round(uv_vis_ind).astype(np.int32)
    X_ind = np.intersect1d(uv_vis_ind, X_vis_ind)

    X = uv_face_xyz[:2, X_ind]
    # filter. only preserve good fits
    thd = 0.02
    good_ind= np.nonzero((X[1,:] - np.round(X[1,:]) < thd) & (X[0,:] - np.round(X[0,:]) < thd))[0]
    X_ind = X_ind[good_ind]

    X = uv_face_xyz[:2, X_ind]
    x = pos_in_uv[np.round(X[1,:]).astype(np.int32), np.round(X[0,:]).astype(np.int32), :2].T    
    #
    x[1,:] = h - 1 - x[1,:]
    x = x.astype(np.float32)
    
    print len(X_ind)
    st = time()
    fitted_pp, fitted_sp, fitted_ep = tdmm.fitting.fit_points_simple(x.astype(np.float32), X_ind, model, max_iter = 4)
    print 'fit time:', time() -st
    
    vertices = mesh.vertices.generate_vertices(model, fitted_sp, fitted_ep)
    projected_vertices = camera.project(vertices, fitted_pp, True)
    projected_vertices[1,:] = h - 1 - projected_vertices[1,:]
    face = image
    face_vertices = projected_vertices

    td = mesh_cython.render.render_texture(face_vertices, mean_texture, light_triangles, 128, 128, 3, True)
    mask = mesh_cython.render.render_texture(face_vertices, np.ones_like(mean_texture), light_triangles, 128, 128, 3)
    face_fitted = face*(1 - mask) + (td*mask + face)*mask
    
    imsave(image_name.replace(image_folder, save_folder).replace('.jpg', '_fitted.jpg'), np.squeeze(face_fitted))
    imsave(image_name.replace(image_folder, save_folder).replace('.jpg', '_shape.jpg'), np.squeeze(td))

    kpt = np.round(face_vertices[:, kpt_ind]).astype(np.int32)
    kpt = np.minimum(np.maximum(kpt, 0), 127)
    # face = face*(1 - mask) + (td*mask + face)*0.8*mask
    face = plot_kpt(face, kpt)
    # np.savetxt(image_name.replace(image_folder, save_folder).replace('.jpg', '.txt'), kpt)
    imsave(image_name.replace(image_folder, save_folder).replace('.jpg', '_kpt.jpg'), np.squeeze(face))

    # print vertices.shape
    # np.savetxt(save_folder+'pc_v.asc', vertices.T)
    # projected_vertices[1:,:] = - projected_vertices[1:,:]
    # projected_vertices[2,:] = projected_vertices[2,:] - np.min(projected_vertices[2,:])
    # np.savetxt(save_folder+'pc_pv.asc', projected_vertices.T)

    z = -projected_vertices[2:,:]
    z = z - np.min(z)
    # print np.min(x), np.max(x)
    # print np.min(z), np.max(z)
    # exit()
    # z = -projected_vertices[2:,:]
    # z = z - np.min(z)
    d = 60.
    z[z > d] = 0
    z = z/d
    depth = mesh_cython.render.render_texture(projected_vertices, z, triangles, h, w, c = 1)
    imsave(image_name.replace(image_folder, save_folder).replace('.jpg', '_depth.jpg'), np.squeeze(depth))
    # imsave(image_name.replace(image_folder, save_folder + 'depth/'), np.squeeze(depth))

