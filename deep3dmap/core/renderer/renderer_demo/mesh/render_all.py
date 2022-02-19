'''
Generate 
Author: YadiraF 
Mail: fengyao@sjtu.edu.cn
Date: 2017/8/27
'''
import numpy as np
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import time
# from numpy.linalg import inv
# from numba import jit

def point_mapping(src_image, src_Pose_para, dst_Pose_para, dst_depth, mapping_type='bilinear'):
	## weak projection
	# src
	[x, y, z, t3dx, t3dy, t3dz, f_src] = src_Pose_para
	t3d = np.array([t3dx, t3dy, t3dz])
	t3d_src = np.tile(t3d, (1, 1)).T
	R_src = get_rotation_matrix(x,y,z)
	# dst
	[x, y, z, t3dx, t3dy, t3dz, f_dst] = dst_Pose_para
	t3d = np.array([t3dx, t3dy, t3dz])
	t3d_dst = np.tile(t3d, (1, 1)).T
	R_dst = get_rotation_matrix(x,y,z)

	# back wrapping
	# dst (x,y) --> src (x, y)
	# V_src = f_src * (1/f_dst) * R_src * R_dst^-1 * (V_dst - t3d_dst) + t3d_src
	[h, w, _] = src_image.shape
	dst_image = np.zeros_like(src_image)
	R_dst_inv = inv(R_dst)
	R = R_src.dot(inv(R_dst))
	for y in range(h):
		#y_dif = np.abs(dst_vertices[1,:] - y)
		#y_ind = (y_dif < 0.1) # bool, the index of near pixels with 2 diff
		#if np.sum(y_ind) == 0:
		#	continue
		#y_ind = np.argwhere(y_ind)
		for x in range(w):
			z = dst_depth[y, x, 0]
			p_dst = np.array([[x, y, z]]).T
			p_src = f_src*(1./f_dst)*R.dot(p_dst - t3d_dst) + t3d_src
			src_texel = p_src[:2, 0]
			if src_texel[0] < 0 or src_texel[0]> w-1 or src_texel[1]<0 or src_texel[1] > h-1:
				continue
			# As the coordinates of the transformed pixel in the image will most likely not lie on a texel, we have to choose how to
			# calculate the pixel colors depending on the next texels
			# there are three different texture interpolation methods: area, bilinear and nearest neighbour

			# nearest neighbour 
			if mapping_type == 'nearest':
				dst_image[y, x, :] = src_image[int(round(src_texel[1])), int(round(src_texel[0])), :]
			# bilinear
			elif mapping_type == 'bilinear':
				# next 4 pixels
				ul = src_image[int(np.floor(src_texel[1])), int(np.floor(src_texel[0])), :]
				ur = src_image[int(np.floor(src_texel[1])), int(np.ceil(src_texel[0])), :]
				dl = src_image[int(np.ceil(src_texel[1])), int(np.floor(src_texel[0])), :]
				dr = src_image[int(np.ceil(src_texel[1])), int(np.ceil(src_texel[0])), :]

				yd = src_texel[1] - np.floor(src_texel[1])
				xd = src_texel[0] - np.floor(src_texel[0])
				dst_image[y, x, :] = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd
	return dst_image	
	

def image_mapping(src_image, src_vertices, triangles, src_triangles_vis, dst_vertices, dst_triangle_buffer, mapping_type = 'nearest'):
	'''
	Args:
		triangles: 3 x ntri

		# src
		src_image: height x width x nchannels
		src_vertices: 3 x nver
		src_triangles_vis: ntri. the visibility of each triangle
		
		# dst
		dst_vertices: 3 x nver
		dst_triangle_buffer: height x width. the triangle index of each pixel in dst image

	Returns:
		dst_image: height x width x nchannels

	'''
	[h, w, c] = src_image.shape
	dst_image = np.zeros_like(src_image)
	for y in range(h):
		for x in range(w):
			tri_ind = dst_triangle_buffer[y,x]
			if tri_ind < 0: # no tri in dst image
				continue 
			#if src_triangles_vis[tri_ind]: # the corresponding triangle in src image is invisible
			#	continue
			
			# then. For this triangle index, map corresponding pixels(in triangles) in src image to dst image
			# Two Methods:
			# M1. Calculate the corresponding affine matrix from src triangle to dst triangle. Then find the corresponding src position of this dst pixel.
			# -- ToDo
			# M2. Calculate the relative position of three vertices in dst triangle, then find the corresponding src position relative to three src vertices.
			tri = triangles[:, tri_ind]
			# dst weight, here directly use the center to approximate because the tri is small
			w0 = w1 = w2 = 1./3
			# src
			src_texel = w0*src_vertices[:, tri[0]] + w1*src_vertices[:, tri[1]] + w2*src_vertices[:, tri[2]] #
			if src_texel[0] < 0 or src_texel[0]> w-1 or src_texel[1]<0 or src_texel[1] > h-1:
				continue
			# As the coordinates of the transformed pixel in the image will most likely not lie on a texel, we have to choose how to
			# calculate the pixel colors depending on the next texels
			# there are three different texture interpolation methods: area, bilinear and nearest neighbour

			# nearest neighbour 
			if mapping_type == 'nearest':
				dst_image[y, x, :] = src_image[int(round(src_texel[1])), int(round(src_texel[0])), :]
			# bilinear
			elif mapping_type == 'bilinear':
				# next 4 pixels
				ul = src_image[int(np.floor(src_texel[1])), int(np.floor(src_texel[0])), :]
				ur = src_image[int(np.floor(src_texel[1])), int(np.ceil(src_texel[0])), :]
				dl = src_image[int(np.ceil(src_texel[1])), int(np.floor(src_texel[0])), :]
				dr = src_image[int(np.ceil(src_texel[1])), int(np.ceil(src_texel[0])), :]

				yd = src_texel[1] - np.floor(src_texel[1])
				xd = src_texel[0] - np.floor(src_texel[0])
				dst_image[y, x, :] = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd

	return dst_image
		
def visOftriangles(vertices, triangles, vertices_vis):
	'''
	Args:
        vertices: 3 x nver
        triangles: 3 x ntri
		vertices_vis: nver. the visibility of each vertex
	Returns:
		triangles_vis: ntri. the visibility of each triangle
	''' 

	triangles_vis = np.zeros(triangles.shape[1], dtype = bool)
	for i in range(triangles.shape[1]):
		tri = triangles[:, i]
		if vertices_vis[tri[0]] & vertices_vis[tri[1]] & vertices_vis[tri[2]]:
			triangles_vis[i] = True
	return triangles_vis


def visOfvertices(image, vertices, triangels, depth_buffer):
	'''
	Args:
        vertices: 3 x nver
        triangles: 3 x ntri
		depth_buffer: height x width
	Returns:
		vertices_vis: nver. the visibility of each vertex
	'''
	[h, w, _] = image.shape
	vertices_vis = np.zeros(vertices.shape[1], dtype = bool)
	for i in range(vertices.shape[1]):
		vertex = vertices[:, i]
		ul = [int(np.floor(vertex[1])), int(np.floor(vertex[0]))]
		ur = [int(np.floor(vertex[1])), int(np.ceil(vertex[0]))]
		dl = [int(np.ceil(vertex[1])), int(np.floor(vertex[0]))]
		dr = [int(np.ceil(vertex[1])), int(np.ceil(vertex[0]))]
		
		#if int(vertex[0]) < 0 o r int(vertex[0]) > h-1 or int(vertex[1]) <0 or int(vertex[1]) > w-1:
		if ul[0] < 0 or dl[0] > h-1 or ul[1] <0 or ur[1] > w-1:
			continue
		
		if (vertex[2] > depth_buffer[ul[0], ul[1]]) & (vertex[2] > depth_buffer[ur[0], ur[1]]) & (vertex[2] > depth_buffer[dl[0], dl[1]]) & (vertex[2] > depth_buffer[dr[0], dr[1]]):
			vertices_vis[i] = True
	return vertices_vis
			
@jit
def z_buffer(image, vertices, triangles):
	'''
	Args:
		image: height x width x nchannels
        vertices: 3 x nver
        triangles: 3 x ntri
	Returns:
		depth_buffer: height x width
		triangle_buffer: height x width
	ToDo:
		whether to add x, y by 0.5? the center of the pixel?
		m3. like somewhere is wrong
    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
	# Here, the bigger the z, the fronter the point.
	'''
	# initial 
	[h, w, _] = image.shape

	depth_buffer = np.zeros([h, w]) + 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
	triangle_buffer = np.zeros_like(depth_buffer, dtype = np.int32) - 1 # if -1, the pixel has no triangle correspondance

    ## calculate the depth(z) of each triangle
	#-m1. z = the center of shpere(through 3 vertices)
	#center3d = (vertices[:, triangles[0,:]] + vertices[:,triangles[1,:]] + vertices[:, triangles[2,:]])/3.
	#tri_depth = np.sum(center3d**2, axis = 0)
	#-m2. z = the center of z(v0, v1, v2)
	tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3.
	
	for i in range(int(triangles.shape[1])):
		tri = triangles[:, i] # 3 vertex indices

		# the inner bounding box
		umin = max(int(np.ceil(np.min(vertices[0,tri]))), 0)
		umax = min(int(np.floor(np.max(vertices[0,tri]))), w-1)

		vmin = max(int(np.ceil(np.min(vertices[1,tri]))), 0)
		vmax = min(int(np.floor(np.max(vertices[1,tri]))), h-1)

		if umax<umin or vmax<vmin:
			continue

		for u in range(umin, umax+1):
			for v in range(vmin, vmax+1):
				#-m3. calculate the accurate depth(z) of each pixel by barycentric weights
				#w0, w1, w2 = weightsOfpoint([u,v], vertices[:2, tri])
				#tri_depth = w0*vertices[2,tri[0]] + w1*vertices[2,tri[1]] + w2*vertices[2,tri[2]]
				if tri_depth[i] < depth_buffer[v, u]: # and is_pointIntri([u,v], vertices[:2, tri]): 
					depth_buffer[v, u] = tri_depth[i]
					triangle_buffer[v, u] = i

	return depth_buffer, triangle_buffer

# --------------------------------------------------------------------------------------------

def triangle2image(src_image, vertex, texture, tri):
    '''
    Args:
        vertex: 3 x nver
        texture: 3 x nver
        tri: 3 x ntri
        image: height x width x nchannels
    '''
    #print vertex.shape, tri.shape, src_image.shape
    nver = vertex.shape[1]
    ntri = tri.shape[1]
    [h, w, c] = src_image.shape

    # for each triangle, have 3 vertex --> 3 points center tex
    tri_points = np.zeros((3, 3, ntri))
    for i in range(3):
        tri_points[:, i, :] = vertex[:, tri[i, :]]

    center3d = (vertex[:, tri[0,:]] + vertex[:,tri[1,:]] + vertex[:, tri[2,:]])/3.
    tri_center_r = np.sum(center3d**2, axis = 0)
    tri_tex = (texture[:, tri[0,:]] + texture[:,tri[1,:]] + texture[:, tri[2,:]])/3.

    # init image
    image = np.zeros_like(src_image)
    imageh = np.zeros((h, w))

    for i in np.arange(ntri):
        umin = int(np.ceil(np.min(tri_points[0, :, i])))
        umax = int(np.floor(np.max(tri_points[0, :, i])))

        vmin = int(np.ceil(np.min(tri_points[1,:,i])))
        vmax = int(np.floor(np.max(tri_points[1,:,i])))

        #print i, tri_points[0,:,i], umin, umax, tri_points[1, :, i], vmin, vmax

        if umax<umin or vmax<vmin or umax>w-1 or umin<0 or vmax>h-1 or vmin<0:
            continue
        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                #print imageh[u,v]
                #print tri_center_z[0,i]
                if imageh[v, u]<tri_center_r[i]: # and is_ptIntri([u,v], tri_points[:2,:,i]):
                    imageh[v, u] = tri_center_r[i]
                    image[v, u, :] = tri_tex[:, i]
    return image

def para2vertices(bfm, para):
	Pose_Para = para[:7]
	Shape_Para = para[7:7+199]
	Exp_Para = para[7+199:]

	# shape vertices	
	vertices = bfm['shapeMU'][:,0] + bfm['shapePC'].dot(Shape_Para) + bfm['w_exp'].dot(Exp_Para)
	vertices = np.resize(vertices, [len(vertices)/3, 3]).T
	
	## weak projection
	[x, y, z, t3dx, t3dy, t3dz, f] = Pose_Para
	t3d = np.array([t3dx, t3dy, t3dz])
	t3d = np.tile(t3d, (vertices.shape[1], 1)).T
	projected_vertices = f * get_rotation_matrix(x, y, z).dot(vertices) + t3d
		
	return projected_vertices



if __name__ == '__main__':
	## load BFM file
	bfm_path='Data/BFM_python.mat'
	bfm=sio.loadmat(bfm_path)
	#bfm['tl'] = bfm['tl'] - 1 # from matlab to python, index-1
	triangles = bfm['tl'].T - 1

	# load image, and fitted parameters	
	name = '000002'
	src_image = imread('Data/' + name + '.jpg')/255.
	C = sio.loadmat('Data/' + name + '.mat')
	src_para = C['para'][0, :]
	# 
	dst_para = np.copy(src_para)
	#dst_para[1] *= 2
	dst_para[:7] = [0, 0, 0, 66, 66, 0, 6e-04]
	#dst_para[7+199:] = np.zeros(29)

	#--- map src image to dst image
	dst_vertices = para2vertices(bfm, dst_para)
	dst_vertices[0,:] -= 1
	dst_vertices[1,:] = src_image.shape[0] + 4 - dst_vertices[1, :]
	start = time.time()
	dst_depth = para2plane(dst_vertices, triangles)
	dst_image_point = point_mapping(src_image, src_para[:7], dst_para[:7], dst_depth) 
	print 'elapsed time:{} s'.format(time.time() - start)

	'''	
	plt.subplot(121)
	plt.imshow(src_image)
	plt.subplot(122)
	plt.imshow(dst_image) # + src_image*(1-dst_mask)*(1-src_mask))
	plt.show()
	'''
	#exit()	

	# 1. get the vertices
	src_vertices = para2vertices(bfm, src_para)
	dst_vertices = para2vertices(bfm, dst_para)
	src_vertices[1,:] = src_image.shape[0] + 4 - src_vertices[1, :]
	dst_vertices[1,:] = src_image.shape[0] + 4 - dst_vertices[1, :]
	#test show
	#plt.imshow(src_image)
	#plt.plot(src_vertices[0,:], src_vertices[1,:], 'r.')
	#plt.show()
	#exit()
	#plt.figure()
	#plt.plot(dst_vertices[0,:], dst_vertices[1,:], 'bx')
	#plt.show()

	# 2. get the needed 
	print triangles.shape
	# for src
	src_depth_buffer, src_triangle_buffer = z_buffer(src_image, src_vertices, triangles)
	#eixt()
	src_vertices_vis = visOfvertices(src_image, src_vertices, triangles, src_depth_buffer)
	src_triangles_vis = visOftriangles(src_vertices, triangles, src_vertices_vis)
	# for dst
	start = time.time()
	_, dst_triangle_buffer = z_buffer(src_image, dst_vertices, triangles)
	print 'elapsed time:{} s'.format(time.time() - start)
	exit()

	# 3. map
	dst_image = image_mapping(src_image, src_vertices, triangles, src_triangles_vis, dst_vertices, dst_triangle_buffer, 'bilinear')
	#dst_image = image_mapping(src_image, src_vertices, triangles, src_triangles_vis, dst_vertices, dst_triangle_buffer)
	elapsed = time.time() - start

	src_mask = (src_triangle_buffer > 0)
	src_mask = np.repeat(np.expand_dims(src_mask, 2), 3, axis = 2)
	dst_mask = (dst_triangle_buffer > 0)
	dst_mask = np.repeat(np.expand_dims(dst_mask, 2), 3, axis = 2)
	
	plt.subplot(131)
	plt.imshow(src_image)
	plt.subplot(132)
	plt.imshow(dst_image_point)
	plt.subplot(133)
	plt.imshow(dst_image) # + src_image*(1-dst_mask)*(1-src_mask))
	plt.show()
	#imsave('Samples/fitting/0_pose/pncc.jpg', pncc)
	#imsave('Samples/fitting/0_pose/depth.jpg', depth)

