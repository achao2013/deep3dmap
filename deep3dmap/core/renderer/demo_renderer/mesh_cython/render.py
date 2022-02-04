import numpy as np
from . import render_cython
from time import time

def get_norm_direction(vertices, triangles):
    pt0 = vertices[:, triangles[0,:]].T
    pt1 = vertices[:, triangles[1,:]].T
    pt2 = vertices[:, triangles[2,:]].T
    tri_norm = np.cross(pt0 - pt1, pt0 - pt2).T

    norm = np.zeros_like(vertices)

    # for i in range(triangles.shape[1]):
    #     norm[:, triangles[0,i]] = norm[:, triangles[0,i]] + tri_norm[:,i]
    #     norm[:, triangles[1,i]] = norm[:, triangles[1,i]] + tri_norm[:,i]
    #     norm[:, triangles[2,i]] = norm[:, triangles[2,i]] + tri_norm[:,i]
    # print 'norm: ', time() - st
    render_cython.get_norm_direction_core(norm, tri_norm.copy(), triangles, vertices.shape[1], triangles.shape[1])

    # normalize to unit length
    mag = np.sum(norm**2, 0)
    zero_ind = (mag == 0)
    mag[zero_ind] = 1;
    norm[0,zero_ind] = np.ones((np.sum(zero_ind)))

    norm = norm/np.sqrt(mag)

    return norm

## ----- Illumination
def fit_illumination(image, vertices, texture, triangles, vis_ind, lamb = 10, max_iter = 3):
    [h, w, c] = image.shape

    # surface normal
    st = time() 
    norm = get_norm_direction(vertices, triangles)
    
    nver = vertices.shape[1]

    # vertices --> corresponding image pixel
    pt2d = vertices[:2, :]

    pt2d[0,:] = np.minimum(np.maximum(pt2d[0,:], 0), w - 1)
    pt2d[1,:] = np.minimum(np.maximum(pt2d[1,:], 0), h - 1)
    pt2d = np.round(pt2d).astype(np.int32) # 2 x nver

    image_pixel = image[pt2d[1,:], pt2d[0,:], :] # nver x 3
    image_pixel = image_pixel.T # 3 x nver

    # vertices --> corresponding mean texture pixel with illumination
    # Spherical Harmonic Basis
    harmonic_dim = 9
    nx = norm[0,:];
    ny = norm[1,:];
    nz = norm[2,:];
    harmonic = np.zeros((nver, harmonic_dim))

    pi = np.pi
    harmonic[:,0] = np.sqrt(1/(4*pi)) * np.ones((nver,));
    harmonic[:,1] = np.sqrt(3/(4*pi)) * nx;
    harmonic[:,2] = np.sqrt(3/(4*pi)) * ny;
    harmonic[:,3] = np.sqrt(3/(4*pi)) * nz;
    harmonic[:,4] = 1/2. * np.sqrt(3/(4*pi)) * (2*nz**2 - nx**2 - ny**2);
    harmonic[:,5] = 3 * np.sqrt(5/(12*pi)) * (ny*nz);
    harmonic[:,6] = 3 * np.sqrt(5/(12*pi)) * (nx*nz);
    harmonic[:,7] = 3 * np.sqrt(5/(12*pi)) * (nx*ny);
    harmonic[:,8] = 3/2. * np.sqrt(5/(12*pi)) * (nx*nx - ny*ny);
    '''
    I' = sum(albedo * lj * hj) j = 0:9 (albedo = tex)
    set A = albedo*h (n x 9)
        alpha = lj (9 x 1)
        Y = I (n x 1)
        Y' = A.dot(alpha)

    opt function:
        ||Y - A*alpha|| + lambda*(alpha'*alpha)
    result:
        A'*(Y - A*alpha) + lambda*alpha = 0
        ==>
        (A'*A*alpha - lambda)*alpha = A'*Y
        left: 9 x 9
        right: 9 x 1
    '''
    n_vis_ind = len(vis_ind)
    n = n_vis_ind*c

    Y = np.zeros((n, 1))
    A = np.zeros((n, 9))
    light = np.zeros((3, 1))

    for k in range(c):
        Y[k*n_vis_ind:(k+1)*n_vis_ind, :] = image_pixel[k, vis_ind][:, np.newaxis]
        A[k*n_vis_ind:(k+1)*n_vis_ind, :] = texture[k, vis_ind][:, np.newaxis] * harmonic[vis_ind, :]
        Ac = texture[k, vis_ind][:, np.newaxis]
        Yc = image_pixel[k, vis_ind][:, np.newaxis]
        light[k] = (Ac.T.dot(Yc))/(Ac.T.dot(Ac))

    for i in range(max_iter):

        Yc = Y.copy()
        for k in range(c):
            Yc[k*n_vis_ind:(k+1)*n_vis_ind, :]  /= light[k]

        # update alpha
        equation_left = np.dot(A.T, A) + lamb*np.eye(harmonic_dim); # why + ?
        equation_right = np.dot(A.T, Yc) 
        alpha = np.dot(np.linalg.inv(equation_left), equation_right)

        # update light
        for k in range(c):
            Ac = A[k*n_vis_ind:(k+1)*n_vis_ind, :].dot(alpha)
            Yc = Y[k*n_vis_ind:(k+1)*n_vis_ind, :]
            light[k] = (Ac.T.dot(Yc))/(Ac.T.dot(Ac))

    appearance = np.zeros_like(texture)
    for k in range(c):
        tmp = np.dot(harmonic*texture[k, :][:, np.newaxis], alpha*light[k])
        appearance[k,:] = tmp.T

    appearance = np.minimum(np.maximum(appearance, 0), 1)

    return appearance

def render_colors(vertices, triangles, colors, h, w, c = 3, isBG = False):
    ''' render mesh with colors by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width    
    '''
    # initial 
    if isBG:
        image = np.ones((h, w, c))
    else:
        image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3. 
    tri_tex = (colors[:, triangles[0,:]] + colors[:,triangles[1,:]] + colors[:, triangles[2,:]])/3.
    
    ###
    render_cython.render_colors_core(
                image, vertices, triangles,
                tri_depth.copy(), tri_tex.copy(), depth_buffer.copy(),
                vertices.shape[1], triangles.shape[1], 
                h, w, c)
    return image


def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, c = 3, mapping_type = 'nearest', isBG = False):
    ''' render mesh with texture map by z buffer
    Args:
        vertices: 3 x nver
        triangles: 3 x ntri
        texture: tex_h x tex_w x 3
        tex_coords: 2 x ntexver
        tex_triangles: 3 x ntri
        h: height of rendering
        w: width of rendering
    '''
    # initial 
    if isBG:
        image = np.ones((h, w, c))
    else:
        image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3. 
    
    tex_h, tex_w, tex_c = texture.shape
    if mapping_type == 'nearest':
        mt = int(0)
    elif mapping_type == 'bilinear':
        mt = int(1)
    else:
        mt = int(0)
    ###
    render_cython.render_texture_core(
                image, vertices, triangles,
                texture, tex_coords, tex_triangles,
                tri_depth.copy(), depth_buffer.copy(),
                vertices.shape[1], tex_coords.shape[0], triangles.shape[1], 
                h, w, c,
                tex_h, tex_w, tex_c,
                mt)
    return image


def map_texture(src_image, src_vertices, dst_vertices, dst_triangle_buffer, triangles, h, w, c = 3, mapping_type = 'bilinear'):
    '''
    Args:
        triangles: 3 x ntri

        # src
        src_image: height x width x nchannels
        src_vertices: 3 x nver
        
        # dst
        dst_vertices: 3 x nver
        dst_triangle_buffer: height x width. the triangle index of each pixel in dst image

    Returns:
        dst_image: height x width x nchannels
    '''
    dst_image = np.zeros((h, w, c))
    render_cython.map_texture_core(dst_image,
                src_image, dst_vertices, src_vertices, 
                dst_triangle_buffer, triangles,
                src_vertices.shape[1], triangles.shape[1], 
                src_image.shape[0], src_image.shape[1], src_image.shape[2],
                h, w, c
                )
    return dst_image



def vis_of_vertices(vertices, triangles, h, w):
    '''
    Args:
        vertices: 3 x nver
        triangles: 3 x ntri
    Returns:
        vertices_vis: nver. the visibility of each vertex
    '''
    vertices_vis = np.zeros(vertices.shape[1])
    
    depth_buffer = np.zeros([h, w]) - 999999.
    depth_tmp = np.zeros_like(depth_buffer) - 9999
    
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3. 
    
    render_cython.vis_of_vertices_core(
                vertices_vis, vertices, triangles,
                tri_depth.copy(), depth_buffer.copy(), depth_tmp.copy(),
                vertices.shape[1], triangles.shape[1], 
                h, w, 1)
    return vertices_vis



def get_triangle_buffer(vertices, triangles, h, w):
    '''
    Args:
        vertices: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width
    Returns:
        depth_buffer: height x width
    ToDo:
        whether to add x, y by 0.5? the center of the pixel?
        m3. like somewhere is wrong
    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # Here, the bigger the z, the fronter the point.
    '''
    # initial 
    depth_buffer = np.zeros([h, w]) - 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
    triangle_buffer = np.zeros_like(depth_buffer, dtype = np.int32) - 1 # if -1, the pixel has no triangle correspondance

    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3. 
    
    render_cython.get_triangle_buffer_core(
                triangle_buffer, vertices, triangles,
                tri_depth.copy(), depth_buffer.copy(),
                vertices.shape[1], triangles.shape[1], 
                h, w, 1)
    return triangle_buffer












def get_correspondence(image, pncc_code):
    nver = pncc_code.shape[1]
    [h, w, c] = image.shape
    uv = np.zeros((2, nver))

    render_cython.get_correspondence_core(image.copy(), pncc_code.copy(), uv, 
                nver,
                h, w, c) 
    X_ind = np.nonzero(uv)[1]
    x = uv[:, X_ind]
    x[1,:] = h - 1 - x[1,:]
    return x, X_ind


# --  Python Version
def get_correspondence_python(image, pncc_code):
    uv_list = []
    verind_list = []
    [h, w, c] = image.shape

    for y in range(int(h)):
        for x in range(int(w)):
            if np.sum(image[y, x, :]) < 0.07:
                continue
            uv_value = image[y, x, :] # 3
            dis = np.sum(np.abs(pncc_code.T - uv_value.T), 1) # nver
            verind = np.argmin(dis)
            if dis[verind] > 0.08:
                continue
            uv_list.append([x,y])
            verind_list.append(verind)
    x = np.array(uv_list).T
    x[1,:] = h - 1 - x[1,:]
    X_ind = np.array(verind_list)
    return x, X_ind