import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
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

from __init__ import *


# --- 1. load model
model = load_model()

# --- 2. generate vertices
import mesh.vertices

sp = mesh.vertices.random_shape_para()
ep = mesh.vertices.random_exp_para()
# ep[:,:] = 0
vertices = mesh.vertices.generate_vertices(model, sp, ep)

# --- 3. project vertices
import geometry.camera as camera
import mesh.texture
triangles = get_triangles(model)


c = 0
for yaw in range(-90, 91, 15):
	s = 2e-03
	rx, ry, rz = [0, yaw*np.pi/180, 0]
	t2d = [20, 120]
	pp = np.array([s, rx, ry, rz, t2d[0], t2d[1]])[:, np.newaxis]
	projected_vertices = camera.project(vertices, pp, True)
	projected_vertices[1,:] = - projected_vertices[1,:]


	h = w = size = 448
	center = [size/2, size/2, 0]
	td = center - np.mean(projected_vertices, 1)
	print td
	projected_vertices = projected_vertices + td[:, np.newaxis]

	tp = mesh.texture.random_tex_para()
	texture = mesh.texture.generate_texture(model, tp)

	import mesh_cython.render
	start = time()
	image = mesh_cython.render.render_texture(projected_vertices, texture, triangles, h, w, 3)
	print 'render, cython vertion, time: ', time() - start

	folder = '/home/fengyao/study/Deep-Image-Analogy/deep_image_analogy/example/_3d2real/A'
	imsave('{}/{}.png'.format(folder,c) , image)
	c += 1
