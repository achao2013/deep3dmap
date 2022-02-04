import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef extern from "render.h":
    void _render_colors_core(
        double* image, double* vertices, int* triangles, 
        double* tri_depth, double* tri_tex, double* depth_buffer,
        int nver, int ntri, 
        int h, int w, int c)

    void _render_texture_core(
        double* image, double* vertices, int* triangles, 
        double* texture, double* tex_coords, int* tex_triangles, 
        double* tri_depth, double* depth_buffer,
        int nver, int tex_nver, int ntri, 
        int h, int w, int c, 
        int tex_h, int tex_w, int tex_c, 
        int mapping_type)

    void _map_texture_core(
        double* dst_image, double* src_image,
        double* dst_vertices, double* src_vertices, 
        int* dst_triangle_buffer, int* triangles,
        int nver, int ntri,
        int sh, int sw, int sc,
        int h, int w, int c)
    
    void _get_correspondence_core(
        double* image, double* pncc_code, 
        double* uv,
        int nver,
        int h, int w, int c)

    void _vis_of_vertices_core(
        double* vis, double* vertices, int* triangles, 
        double* tri_depth, double* depth_buffer, double* depth_tmp, 
        int nver, int ntri,
        int h, int w, int c)
    
    void _get_triangle_buffer_core(
        int* triangle_buffer, double* vertices, int* triangles, 
        double* tri_depth, double* depth_buffer,
        int nver, int ntri,
        int h, int w, int c)

    void _get_norm_direction_core(
        double* norm, double* tri_norm, int* triangles,
        int nver, int ntri)

def get_norm_direction_core(np.ndarray[double, ndim=2, mode = "c"] norm not None, 
                np.ndarray[double, ndim=2, mode = "c"] tri_norm not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                int nver, int ntri
        ):
    _get_norm_direction_core(
        <double*> np.PyArray_DATA(norm), <double*> np.PyArray_DATA(tri_norm), <int*> np.PyArray_DATA(triangles),  
        nver, ntri)

def render_colors_core(np.ndarray[double, ndim=3, mode = "c"] image not None, 
                np.ndarray[double, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[double, ndim=1, mode = "c"] tri_depth not None,
                np.ndarray[double, ndim=2, mode = "c"] tri_tex not None,
                np.ndarray[double, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int ntri,
                int h, int w, int c
                ):   
    _render_colors_core(
        <double*> np.PyArray_DATA(image), <double*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <double*> np.PyArray_DATA(tri_depth), <double*> np.PyArray_DATA(tri_tex), <double*> np.PyArray_DATA(depth_buffer),
        nver, ntri,
        h, w, c)

def render_texture_core(np.ndarray[double, ndim=3, mode = "c"] image not None, 
                np.ndarray[double, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[double, ndim=3, mode = "c"] texture not None, 
                np.ndarray[double, ndim=2, mode = "c"] tex_coords not None, 
                np.ndarray[int, ndim=2, mode="c"] tex_triangles not None, 
                np.ndarray[double, ndim=1, mode = "c"] tri_depth not None,
                np.ndarray[double, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int tex_nver, int ntri,
                int h, int w, int c,
                int tex_h, int tex_w, int tex_c,
                int mapping_type
                ):   
    _render_texture_core(
        <double*> np.PyArray_DATA(image), <double*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <double*> np.PyArray_DATA(texture), <double*> np.PyArray_DATA(tex_coords), <int*> np.PyArray_DATA(tex_triangles),  
        <double*> np.PyArray_DATA(tri_depth), <double*> np.PyArray_DATA(depth_buffer),
        nver, tex_nver, ntri,
        h, w, c, 
        tex_h, tex_w, tex_c, 
        mapping_type)


def map_texture_core(np.ndarray[double, ndim=3, mode = "c"] dst_image not None, 
                np.ndarray[double, ndim=3, mode = "c"] src_image not None, 
                np.ndarray[double, ndim=2, mode = "c"] dst_vertices not None, 
                np.ndarray[double, ndim=2, mode = "c"] src_vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] dst_triangle_buffer not None, 
                np.ndarray[int, ndim=2, mode = "c"] triangles not None,
                int nver, int ntri,
                int sh, int sw, int sc,
                int h, int w, int c
                ): 

    _map_texture_core(
        <double*> np.PyArray_DATA(dst_image), <double*> np.PyArray_DATA(src_image), 
        <double*> np.PyArray_DATA(dst_vertices), <double*> np.PyArray_DATA(src_vertices), 
        <int*> np.PyArray_DATA(dst_triangle_buffer), <int*> np.PyArray_DATA(triangles),
        nver, ntri,
        sh, sw, sc,
        h, w, c);


def get_correspondence_core(np.ndarray[double, ndim=3, mode = "c"] image not None, 
                np.ndarray[double, ndim=2, mode = "c"] pncc_code not None, 
                np.ndarray[double, ndim=2, mode = "c"] uv not None, 
                int nver,
                int h, int w, int c
                ):   
    _get_correspondence_core(
        <double*> np.PyArray_DATA(image), <double*> np.PyArray_DATA(pncc_code), 
        <double*> np.PyArray_DATA(uv),
        nver,
        h, w, c)

def vis_of_vertices_core(np.ndarray[double, ndim=1, mode = "c"] vis not None,
                np.ndarray[double, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[double, ndim=1, mode = "c"] tri_depth not None,
                np.ndarray[double, ndim=2, mode = "c"] depth_buffer not None,
                np.ndarray[double, ndim=2, mode = "c"] depth_tmp not None,
                int nver, int ntri,
                int h, int w, int c
                ):   
    _vis_of_vertices_core(
        <double*> np.PyArray_DATA(vis), <double*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <double*> np.PyArray_DATA(tri_depth), <double*> np.PyArray_DATA(depth_buffer), <double*> np.PyArray_DATA(depth_tmp),
        nver, ntri,
        h, w, c)

def get_triangle_buffer_core(np.ndarray[int, ndim=2, mode = "c"] triangle_buffer not None,
                np.ndarray[double, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[double, ndim=1, mode = "c"] tri_depth not None,
                np.ndarray[double, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int ntri,
                int h, int w, int c
                ):   
    _get_triangle_buffer_core(
        <int*> np.PyArray_DATA(triangle_buffer), <double*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <double*> np.PyArray_DATA(tri_depth), <double*> np.PyArray_DATA(depth_buffer),
        nver, ntri,
        h, w, c)



# cdef np.ndarray[np.float32_t, ndim=3] _render(np.ndarray[np.float32_t, ndim=2] vertices, np.ndarray[np.int32_t, ndim=2] triangles, np.ndarray[np.float32_t, ndim=1] tri_depth, np.ndarray[np.float32_t, ndim=2] tri_tex, np.ndarray[np.float32_t, ndim=2] depth_buffer):
#     cdef np.ndarray[np.float32_t, ndim=3] image
#     cdef int ntri
#     cdef int umin, umax, vmin, vmax
#     cdef np.ndarray[np.int32_t, ndim=1] tri
#     cdef float ver00, ver01, ver02, ver10, ver11, ver12
#     #cdef np.ndarray[np.float32_t, ndim=1] ver
#     cdef int w, h

#     w = h = 128
#     ntri = triangles.shape[1]
#     image = np.zeros((w, h, 3), dtype = np.float32)

#     cdef int i, u, v
#     for i from 0 <= i < ntri:
#         tri = triangles[:, i] # 3 vertex indices
#         ver00, ver01, ver02 = vertices[0, tri]
#         ver10, ver11, ver12 = vertices[1, tri]
#         # the inner bounding box
#         umin = _max_int(ceil(_min(ver00, ver01, ver02)), 0)
#         umax = _min_int(floor(_max(ver00, ver01, ver02)), w-1)

#         vmin = _max_int(ceil(_min(ver10, ver11, ver12)), 0)
#         vmax = _min_int(floor(_max(ver10, ver11, ver12)), h-1)

#         if umax<umin or vmax<vmin:
#             continue
#         for u from umin <= u < umax+1:
#             for v from vmin <= v < vmax+1:
#                 if tri_depth[i] > depth_buffer[v, u]: # and is_pointIntri([u,v], vertices[:2, tri]): 
#                     depth_buffer[v, u] = tri_depth[i]
#                     image[v, u, :] = tri_tex[:, i]
#     return image



# def render_texture(vertices, texture, triangles, h, w):
#     '''
#     Args:
#         vertices: 3 x nver
#         texture: 3 x nver
#         triangles: 3 x ntri
#         h: height
#         w: width    
#     '''
#     # initial 
#     image = np.zeros((h, w, 3))

#     depth_buffer = np.zeros([h, w], dtype = np.float32) - 999999.
#     # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
#     tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3. 
#     tri_tex = (texture[:, triangles[0,:]] + texture[:,triangles[1,:]] + texture[:, triangles[2,:]])/3.
#     image = _render(vertices, triangles, tri_depth, tri_tex, depth_buffer)
#     return image