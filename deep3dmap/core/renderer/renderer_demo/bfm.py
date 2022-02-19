import numpy as np
import scipy.io as sio
from scipy.misc import imread, imsave, imrotate
from time import time

import geometry
import geometry.camera
import mesh
import mesh.vertices
import mesh.texture
import mesh.render
import mesh_cython
import mesh_cython.render

import fitting

np.set_printoptions(suppress=True)


class BFM:
    def  __init__(self):
        self.load_model()


    def load_model(self, model_name = 'Data/BFM.mat'):
        ''' load 3DMM model
        Returns:
            model: (nver(n) = 53215, ntri = 105840)
                'shapeMU': 3n x 1
                'shapePC': 3n x 199
                'shapeEV': 199 x 1
                'expMU': 3n x 1
                'expPC': 3n x 29
                'expEV': 29 x 1
                'texMU': 3n x 1
                'texPC': 3n x 199
                'texEV': 199 x 1
                'tri': 3 x ntri (from 1, should sub 1 in python and c++)
                'kpt_ind': 1 x 68 (from 1)
                'tri_mouth': 3 x 114 (from 1, as a supplement to mouth triangles)
        '''
        C = sio.loadmat(model_name)
        model = C['model']
        model = model[0,0]
        # process from matlab style to python/c style 
        # model['shapeMU'] = model['shapeMU'].copy(order='C')

        # change dtype from double(np.float64) to np.float32, 
        # since big matrix process(espetially matrix dot) is too slow in python.
        model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
        model['shapePC'] = model['shapePC'].astype(np.float32)
        model['shapeEV'] = model['shapeEV'].astype(np.float32)
        model['expEV'] = model['expEV'].astype(np.float32)
        model['expPC'] = model['expPC'].astype(np.float32)

        # matlab start with 1
        model['tri'] = model['tri'].copy(order = 'C').astype(np.int32) - 1
        model['tri_mouth'] = model['tri_mouth'].copy(order = 'C').astype(np.int32) - 1
        
        self.model = model

    def load_model_info(model_name = 'Data/BFM_info.mat'):
        ''' load 3DMM model information
        Returns:  (all vertex ind need to sub 1 for python use)
            model_info:
                'symlist': 2 x 26720
                'symlist_tri': 2 x 52937
                'segbin': 4 x n (0: nose, 1: eye, 2: mouth, 3: cheek)
                'segbin_tri': 4 x ntri 
                'face_contour': 1 x 28
                'face_contour_line': 1 x 512
                'face_contour_front': 1 x 28
                'face_contour_front_line': 1 x 512
                'nose_hole': 1 x 142
                'nose_hole_right': 1 x 71
                'nose_hole_left': 1 x 71
                'parallel': 17 x 1 cell
                'parallel_face_contour': 28 x 1 cell

                'uv_coords': 2 x n
        '''
        C = sio.loadmat(model_name)
        model_info = C['model_info']
        model_info = model_info[0,0]
        model_info['uv_coords'] = model_info['uv_coords'].copy(order = 'C')
        self.model_info = model_info

    @property
    def model_num(self):
        nver = self.model['shapePC'].shape[0]/3
        ntri = self.model['tri'].shape[1]
        return nver, ntri
    
    @property
    def para_num(self):
        n_shape_para = self.model['shapePC'].shape[1]
        n_exp_para = self.model['expPC'].shape[1]
        return n_shape_para, n_exp_para

    @property
    def kpt_ind(self):
        ''' get 68 keypoints index
        '''
        kpt_ind = np.squeeze(self.model['kpt_ind']) - 1
        return kpt_ind.astype(np.int32)

    @property
    def organ_ind(self):
        ''' get nose, eye, mouth index
        '''
        valid_bin = self.model_info['segbin'].astype(bool)
        # nver = valid_bin.shape[1]
        organ_ind = np.nonzero(valid_bin[0,:])[0]
        for i in range(1, valid_bin.shape[0] - 1):
            organ_ind = np.union1d(organ_ind, np.nonzero(valid_bin[i,:])[0])
        return organ_ind.astype(np.int32)


    def get_triangles(self, isFull = True):
        ''' get mesh triangles
        Returns:
            triangels: (3, ntri)
        '''
        triangles = model['tri']
        if isFull:
            triangles = np.hstack((triangles, model['tri_mouth']))
        return triangles

    def load_pncc_code(self, name = 'Data/pncc_code.mat'):
        C = sio.loadmat(name)
        pncc_code = C['vertex_code']
        return pncc_code



# Example 1
def pipline():
 # ----------- Forward: parameters(pose, shape, expression) -->  2D image points ------
    
    # --- 1. load model
    model = load_model()

    # --- 2. generate vertices
    import mesh.vertices

    sp = mesh.vertices.random_shape_para()
    ep = mesh.vertices.random_exp_para()
    vertices = mesh.vertices.generate_vertices(model, sp, ep)

    # --- 3. project vertices
    import geometry.camera as camera

    s = 2e-04
    rx, ry, rz = [0, 0, 0]
    
    t2d = [20, 120]

    pp = np.array([s, rx, ry, rz, t2d[0], t2d[1]])[:, np.newaxis]
    projected_vertices = camera.project(vertices, pp, True)

    # ---- 4. useful infomation
    triangles = get_triangles(model)
    kpt_ind = get_kpt_ind(model)

    # ------------ show 
    # plt.subplot(1,2,1)
    # plt.plot(vertices[0,:], vertices[1,:], 'bx')
    # plt.plot(vertices[0,kpt_ind], vertices[1,kpt_ind], 'r.')

    # plt.subplot(1,2,2)
    # plt.plot(projected_vertices[0,:], projected_vertices[1,:], 'bx')
    # plt.plot(projected_vertices[0,kpt_ind], projected_vertices[1,kpt_ind], 'r.')
    # plt.show()


    # ----------- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
    # from fit_points import fit_points

    # x = projected_vertices[:2, kpt_ind]
    # X_ind = kpt_ind
    
    # start = time()
    # fitted_pp, fitted_sp, fitted_ep = fit_points(x, X_ind, model, max_iter = 4)
    # print 'fitting with 68 keypoints, time: ', time() - start
    # print 'pose, groudtruth: \n', pp.flatten()
    # print 'pose, fitted: \n', fitted_pp.flatten()


    # ----------- With texture (texture for color in vertices, uv_texture for triangle texture)
    import mesh.texture
    # 3 texture
    #1. color
    tp = mesh.texture.random_tex_para()
    texture = mesh.texture.generate_texture(model, tp)
    #2. depth
    # depth = np.tile(projected_vertices[2:, :], [3,1])
    #3. pncc
    # pncc = get_pncc_code()

    # --------------  render (have a c++ version in mesh_cython, python version is too slow)
    h = w = 128
    c = 3
    projected_vertices[1,:] = h - 1 - projected_vertices[1,:]

    # import mesh.render
    # start = time()
    # image = mesh.render.render_texture(projected_vertices, texture, triangles, h, w, c)
    # print 'render, python vertion, time: ', time() - start
    # imsave('test_python.jpg', image)


    import mesh_cython.render
    start = time()
    image = mesh_cython.render.render_texture(projected_vertices, texture, triangles, h, w)
    print 'render, cython vertion, time: ', time() - start
    imsave('test_cython.jpg', image)



if __name__ == '__main__':

   

