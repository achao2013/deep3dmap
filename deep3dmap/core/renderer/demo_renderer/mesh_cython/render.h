#ifndef RENDER_HPP_
#define RENDER_HPP_

#include <stdio.h>
#include <cmath>
#include <algorithm>  

using namespace std;

class point
{
 public:
    double x;
    double y;

    double dot(point p)
    {
        return this->x * p.x + this->y * p.y;
    }

    point operator-(const point& p)
    {
        point np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    }

    point operator+(const point& p)
    {
        point np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    }

    point operator*(double s)
    {
        point np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    }
}; 


bool isPointInTri(point p, point p0, point p1, point p2, int h, int w);
void get_point_weight(double* weight, point p, point p0, point p1, point p2);

void _get_norm_direction_core(
    double* norm, double* tri_norm, int* triangles,
    int nver, int ntri);

void _render_colors_core(
    double* image, double* vertices, int* triangles, 
    double* tri_depth, double* tri_tex, double* depth_buffer,
    int nver, int ntri,
    int h, int w, int c);

void _render_texture_core(
    double* image, double* vertices, int* triangles, 
    double* texture, double* tex_coords, int* tex_triangles, 
    double* tri_depth, double* depth_buffer,
    int nver, int tex_nver, int ntri, 
    int h, int w, int c, 
    int tex_h, int tex_w, int tex_c, 
    int mapping_type);

void _vis_of_vertices_core(
    double* vis, double* vertices, int* triangles, 
    double* tri_depth, double* depth_buffer, double* depth_tmp,
    int nver, int ntri,
    int h, int w, int c);

void _get_triangle_buffer_core(
    int* triangle_buffer, double* vertices, int* triangles, 
    double* tri_depth, double* depth_buffer,
    int nver, int ntri,
    int h, int w, int c);

void _get_correspondence_core(
    double* image, double* pncc_code, 
    double* uv,
    int nver,
    int h, int w, int c);

void _map_texture_core(
    double* dst_image, double* src_image,
    double* dst_vertices, double* src_vertices, 
    int* dst_triangle_buffer, int* triangles,
    int nver, int ntri,
    int sh, int sw, int sc,
    int h, int w, int c);


// void _write_obj(std::string filename,
//     double* vertices, double* uv_vertices, double* triangles,
//     int nver, int ntri);
#endif