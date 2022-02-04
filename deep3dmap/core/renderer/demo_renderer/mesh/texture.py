'''
Generate vertices
Estimating parameters about vertices: shape para, exp para
'''

import numpy as np 
from time import time
#-------  generate texture: from 3DMM color
def random_tex_para(n_tex_para = 199):
    tp = np.random.rand(n_tex_para, 1)
    #C = sio.loadmat('Data/para.mat')
    #sp = C['alpha']
    return tp

def generate_texture(model, tex_para = np.zeros((199, 1))):
    '''
    Args:
        model: 3DMM model
        tex_para: (199, 1)
    Returns:
        vertices: (3, nver)
    '''
    texture = model['texMU'] + model['texPC'].dot(tex_para*model['texEV'])
    texture = np.reshape(texture, [3, len(texture)/3], 'F')/255.  # pay attention the order changes(C--> F style)
    
    return texture

# ------ extract texture from image




# ------ map texture from src image to dst image 
def get_norm_direction(vertices, triangles):
    pt0 = vertices[:, triangles[0,:]].T
    pt1 = vertices[:, triangles[1,:]].T
    pt2 = vertices[:, triangles[2,:]].T
    tri_norm = np.cross(pt0 - pt1, pt0 - pt2).T

    norm = np.zeros_like(vertices)
    st = time()

    for i in range(triangles.shape[1]):
        norm[:, triangles[0,i]] = norm[:, triangles[0,i]] + tri_norm[:,i]
        norm[:, triangles[1,i]] = norm[:, triangles[1,i]] + tri_norm[:,i]
        norm[:, triangles[2,i]] = norm[:, triangles[2,i]] + tri_norm[:,i]
    
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

