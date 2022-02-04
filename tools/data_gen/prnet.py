import numpy as np
import os, sys
sys.path.append('.')
import scipy

from skimage.transform import estimate_transform, warp
import cv2
# from imageio import imread, imsave
# from cv2 import imwrite
from glob import glob
import scipy.io as sio
from time import time
import argparse
import ast

import deep3dmap.core.renderer.demo_renderer as demo_renderer
import deep3dmap.core.renderer.demo_renderer.geometry.camera as camera
import deep3dmap.core.renderer.demo_renderer.mesh as mesh
import deep3dmap.core.renderer.demo_renderer.mesh_cython as mesh_cython

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1

class GeneratePos:
    '''
    '''
    def __init__(self, resolution_inp, resolution, 
        model_path = os.path.join('magicbox/face/BFM.mat'), model_info_path = os.path.join('magicbox/face/BFM_info.mat')):
        
        self.resolution_inp = resolution_inp
        self.resolution = resolution
        # load model
        self.model = demo_renderer.load_model(model_path)
        # model information
        model_info = demo_renderer.load_model_info(model_info_path)
        uv_coords = model_info['uv_coords']
        nver, ntri = demo_renderer.get_model_num(self.model)
        projected_vertices = np.vstack((uv_coords*(self.resolution - 1), np.zeros((1, nver))))
        projected_vertices[1,:] = self.resolution - 1 - projected_vertices[1,:]
        self.uv_vertices = projected_vertices.copy()

        self.triangles = demo_renderer.get_triangles(self.model)
        # n_shape_para, n_exp_para = demo_renderer.get_para_num(model)
        # n_tex_para = n_shape_para
        self.kpt_ind = demo_renderer.get_kpt_ind(self.model)

    def detect_kpt(self, kpt):
        # kpt: 3x68
        left = np.min(kpt[0, :])
        right = np.max(kpt[0, :])
        top = np.min(kpt[1,:])
        bottom = np.max(kpt[1,:])
        center = np.array([right - (right - left) / 2.0, 
                 bottom - (bottom - top) / 2.0])
        old_size = (right - left + bottom - top)/2
        size = int(old_size*1.5)

        # random pertube
        marg = old_size*0.1
        t_x = np.random.rand()*marg*2 - marg
        t_y = np.random.rand()*marg*2 - marg
        center[0] = center[0]+t_x; center[1] = center[1]+t_y

        size = size*(np.random.rand()*0.2 + 0.9)
        return center, size

    def process(self, image_path, save_folder):
        image_name = image_path.strip().split('/')[-1]
        save_path = os.path.join(save_folder, image_name)

        # 1. load image and fitted parameters
        #image = imread(image_path)#/255.
        image = cv2.imread(image_path)/255.0
        image=image[:,:,[2,1,0]]
        [h, w, c] = image.shape

        mat_path = image_path.replace('jpg', 'mat')
        info = sio.loadmat(mat_path)
        pose_para = info['Pose_Para'].T.astype(np.float32)
        shape_para = info['Shape_Para'].astype(np.float32)
        exp_para = info['Exp_Para'].astype(np.float32)

        # 2. generate mesh
        vertices = mesh.vertices.generate_vertices(self.model, shape_para, exp_para)
        # camera --> project vertices
        projected_vertices = camera.project_3ddfa_128(vertices, pose_para, True)
        projected_vertices[1,:] = h - 1 - projected_vertices[1,:]
        
        # # validate
        kpt = projected_vertices[:, self.kpt_ind].astype(np.int32)
        # imsave(save_path, plot_kpt(image/255., kpt))
        
        # 3. crop image
        # if self.detect_faces(image) is not None:
        #     center, size = self.detect_faces(image)
        # else:
        center, size = self.detect_kpt(kpt)

        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        
        vertices = projected_vertices.copy()
        vertices[2,:] = 1
        vertices = np.dot(tform.params, vertices)
        vertices[2,:] = projected_vertices[2,:]*tform.params[0,0]
        vertices[2,:] = vertices[2,:] - np.min(vertices[2,:])

        # 4. UV position map      
        uv_pos = mesh_cython.render.render_colors(self.uv_vertices, self.triangles, vertices, h = self.resolution, w = self.resolution)

        # 5. save files
        #imsave(save_path.replace('.jpg','_inp.jpg'), cropped_image)
        cv2.imwrite(save_path.replace('.jpg','_inp.jpg'), np.rint(cropped_image[:,:,[2,1,0]]*255))
        np.save(save_path.replace('.jpg', '.npy'), uv_pos[:, :, :])
        # imsave(save_path.replace('.jpg', '_pos.jpg'), np.round(uv_pos/self.resolution*255).astype(np.uint8)) #*uv_mask[:,:,np.newaxis] )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data')

    parser.add_argument('-i', '--inputDir', default='Data/300W_LP', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='Data/300W_LP_256', type=str,
                        help='path to the output directory, where results will be saved.')

    image_folder = parser.parse_args().inputDir
    save_folder = parser.parse_args().outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    resolution_inp = 256 
    resolution = 256
    gp = GeneratePos(resolution_inp, resolution)
    
    image_path_list = glob(os.path.join(image_folder, '*.jpg'))
    total_num = len(image_path_list)
    st = time()
    for i, image_path in enumerate(image_path_list):
        if i%1000 == 0:
            print('processed {}/{}; time: {}min'.format(i, total_num, (time() - st)/60))
        gp.process(image_path, save_folder)
        #if i>10:
        #    break
