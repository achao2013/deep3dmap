#include "render.h"


void _get_norm_direction_core(
    double* norm, double* tri_norm, int* triangles,
    int nver, int ntri)
{
    int i, j;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;

    for(i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[i];
        tri_p1_ind = triangles[ntri + i];
        tri_p2_ind = triangles[2*ntri + i]; 

        for(j = 0; j < 3; j++)
        {
            norm[j*nver + tri_p0_ind] = norm[j*nver + tri_p0_ind] + tri_norm[j*ntri + i];
            norm[j*nver + tri_p1_ind] = norm[j*nver + tri_p1_ind] + tri_norm[j*ntri + i];
            norm[j*nver + tri_p2_ind] = norm[j*nver + tri_p2_ind] + tri_norm[j*ntri + i];
        }
    }
}


void _render_colors_core(
    double* image, double* vertices, int* triangles, 
    double* tri_depth, double* tri_tex, double* depth_buffer,
    int nver, int ntri,
    int h, int w, int c)
{
    int i;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;

    for(i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[i];
        tri_p1_ind = triangles[ntri + i];
        tri_p2_ind = triangles[2*ntri + i];

        // printf("tri_ind: %d, %d, %d, %d \n", i, tri_v0_ind, tri_v1_ind, tri_v2_ind);

        // v0_x = vertices[tri_v0_ind]; v0_y = vertices[nver + tri_v0_ind];
        // v1_x = vertices[tri_v1_ind]; v1_y = vertices[nver + tri_v1_ind];
        // v2_x = vertices[tri_v2_ind]; v2_y = vertices[nver + tri_v2_ind];
        // x_min = max((int)ceil(min(v0_x, min(v1_x, v2_x))), 0);
        // x_max = min((int)floor(max(v0_x, max(v1_x, v2_x))), w - 1);
      
        // y_min = max((int)ceil(min(v0_y, min(v1_y, v2_y))), 0);
        // y_max = min((int)floor(max(v0_y, max(v1_y, v2_y))), h - 1);


        p0.x = vertices[tri_p0_ind]; p0.y = vertices[nver + tri_p0_ind];
        p1.x = vertices[tri_p1_ind]; p1.y = vertices[nver + tri_p1_ind];
        p2.x = vertices[tri_p2_ind]; p2.y = vertices[nver + tri_p2_ind];

        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
      
        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);


        if(x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for(y = y_min; y <= y_max; y++) //h
        {
            for(x = x_min; x <= x_max; x++) //w
            {
                p.x = x; p.y = y;
                if((tri_depth[i] > depth_buffer[y*w + x]) && isPointInTri(p, p0, p1, p2, h, w))
                {
                    depth_buffer[y*w + x] = tri_depth[i];
                    for(k = 0; k < c; k++)
                    {
                        image[y*w*c + x*c + k] = tri_tex[k*ntri + i];
                    }
                }
            }
        }
    }
}


void _render_texture_core(
    double* image, double* vertices, int* triangles, 
    double* texture, double* tex_coords, int* tex_triangles, 
    double* tri_depth, double* depth_buffer,
    int nver, int tex_nver, int ntri, 
    int h, int w, int c, 
    int tex_h, int tex_w, int tex_c, 
    int mapping_type)
{
    int i;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    int tex_tri_p0_ind, tex_tri_p1_ind, tex_tri_p2_ind;
    point p0, p1, p2, p;
    point tex_p0, tex_p1, tex_p2, tex_p;
    int x_min, x_max, y_min, y_max;
    double weight[3];
    double xd, yd;
    double ul, ur, dl, dr;
    for(i = 0; i < ntri; i++)
    {
        // mesh
        tri_p0_ind = triangles[i];
        tri_p1_ind = triangles[ntri + i];
        tri_p2_ind = triangles[2*ntri + i];

        p0.x = vertices[tri_p0_ind]; p0.y = vertices[nver + tri_p0_ind];
        p1.x = vertices[tri_p1_ind]; p1.y = vertices[nver + tri_p1_ind];
        p2.x = vertices[tri_p2_ind]; p2.y = vertices[nver + tri_p2_ind];

        // texture
        tex_tri_p0_ind = tex_triangles[i];
        tex_tri_p1_ind = tex_triangles[ntri + i];
        tex_tri_p2_ind = tex_triangles[2*ntri + i];

        tex_p0.x = tex_coords[tex_tri_p0_ind]; tex_p0.y = tex_coords[tex_nver + tri_p0_ind];
        tex_p1.x = tex_coords[tex_tri_p1_ind]; tex_p1.y = tex_coords[tex_nver + tri_p1_ind];
        tex_p2.x = tex_coords[tex_tri_p2_ind]; tex_p2.y = tex_coords[tex_nver + tri_p2_ind];


        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
      
        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);


        if(x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for(y = y_min; y <= y_max; y++) //h
        {
            for(x = x_min; x <= x_max; x++) //w
            {
                p.x = x; p.y = y;
                if((tri_depth[i] > depth_buffer[y*w + x]) && isPointInTri(p, p0, p1, p2, h, w))
                {
                    // -- color from texture
                    // cal weight in mesh tri
                    get_point_weight(weight, p, p0, p1, p2);
                    // cal coord in texture
                    tex_p = tex_p0*weight[0] + tex_p1*weight[1] + tex_p2*weight[2];

                    yd = tex_p.y - floor(tex_p.y);
                    xd = tex_p.x - floor(tex_p.x);
                    for(k = 0; k < c; k++)
                    {
                        if(mapping_type==0)// nearest
                        {   
                            image[y*w*c + x*c + k] = texture[int(round(tex_p.y))*tex_w*tex_c + int(round(tex_p.x))*tex_c + k];
                        }
                        else//bilinear interp
                        { 
                            ul = texture[(int)floor(tex_p.y)*tex_w*tex_c + (int)floor(tex_p.x)*tex_c + k];
                            ur = texture[(int)floor(tex_p.y)*tex_w*tex_c + (int)ceil(tex_p.x)*tex_c + k];
                            dl = texture[(int)ceil(tex_p.y)*tex_w*tex_c + (int)floor(tex_p.x)*tex_c + k];
                            dr = texture[(int)ceil(tex_p.y)*tex_w*tex_c + (int)ceil(tex_p.x)*tex_c + k];

                            image[y*w*c + x*c + k] = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd;
                        }
                    }


                    // -- depth buffer
                    depth_buffer[y*w + x] = tri_depth[i];

                }
            }
        }
    }
}


void _map_texture_core(
    double* dst_image, double* src_image,
    double* dst_vertices, double* src_vertices, 
    int* dst_triangle_buffer, int* triangles,
    int nver, int ntri,
    int sh, int sw, int sc,
    int h, int w, int c)
{
    int tri_ind;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    point p, p0, p1, p2, src_texel;
    // double ul[c], ur[c], dl[c], dr[c];
    double xd, yd;
    double ul, ur, dl, dr;
    double weight[3];

    for(y = 0; y < h; y++)
    {
        for(x = 0; x < w; x++)
        {
            tri_ind = dst_triangle_buffer[y*w + x];
            if(tri_ind < 0)
                continue;
            tri_p0_ind = triangles[tri_ind];
            tri_p1_ind = triangles[ntri + tri_ind];
            tri_p2_ind = triangles[2*ntri + tri_ind];

            p.x = x; p.y = y;
            p0.x = dst_vertices[tri_p0_ind]; p0.y = dst_vertices[nver + tri_p0_ind];
            p1.x = dst_vertices[tri_p1_ind]; p1.y = dst_vertices[nver + tri_p1_ind];
            p2.x = dst_vertices[tri_p2_ind]; p2.y = dst_vertices[nver + tri_p2_ind];
            
            get_point_weight(weight, p, p0, p1, p2);

            p0.x = src_vertices[tri_p0_ind]; p0.y = src_vertices[nver + tri_p0_ind];
            p1.x = src_vertices[tri_p1_ind]; p1.y = src_vertices[nver + tri_p1_ind];
            p2.x = src_vertices[tri_p2_ind]; p2.y = src_vertices[nver + tri_p2_ind];
            
            // weight[0] = weight[1] = weight[2] = 1./3;
            src_texel = p0*weight[0] + p1*weight[1] + p2*weight[2];

            if(src_texel.x < 0 || src_texel.x > sw-1 || src_texel.y < 0 || src_texel.y > sh-1)
                continue;

            yd = src_texel.y - floor(src_texel.y);
            xd = src_texel.x - floor(src_texel.x);
            for(k = 0; k < c; k++)
            {
                // nearest
                // dst_image[y*w*c + x*c + k] = src_image[int(round(src_texel.y))*sw*sc + int(round(src_texel.x))*sc + k];
                
                // bilinearinterp
                ul = src_image[(int)floor(src_texel.y)*sw*sc + (int)floor(src_texel.x)*sc + k];
                ur = src_image[(int)floor(src_texel.y)*sw*sc + (int)ceil(src_texel.x)*sc + k];
                dl = src_image[(int)ceil(src_texel.y)*sw*sc + (int)floor(src_texel.x)*sc + k];
                dr = src_image[(int)ceil(src_texel.y)*sw*sc + (int)ceil(src_texel.x)*sc + k];

                dst_image[y*w*c + x*c + k] = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd;
            }

        }
    }
}


void _vis_of_vertices_core(
    double* vis, double* vertices, int* triangles, 
    double* tri_depth, double* depth_buffer, double* depth_tmp, 
    int nver, int ntri,
    int h, int w, int c)
{
    int i;
    int x, y, z;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;

    for(i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[i];
        tri_p1_ind = triangles[ntri + i];
        tri_p2_ind = triangles[2*ntri + i];

        p0.x = vertices[tri_p0_ind]; p0.y = vertices[nver + tri_p0_ind];
        p1.x = vertices[tri_p1_ind]; p1.y = vertices[nver + tri_p1_ind];
        p2.x = vertices[tri_p2_ind]; p2.y = vertices[nver + tri_p2_ind];

        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
      
        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);


        if(x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for(y = y_min; y <= y_max; y++) //h
        {
            for(x = x_min; x <= x_max; x++) //w
            {
                p.x = x; p.y = y;
                if((tri_depth[i] > depth_buffer[y*w + x]) && isPointInTri(p, p0, p1, p2, h, w))
                {
                    depth_buffer[y*w + x] = tri_depth[i];
                }
            }
        }
    }

    for(i = 0; i < nver; i++)
    {
        x = vertices[i]; 
        y = vertices[nver + i];
        if(floor(x) < 0 || ceil(x) > w-1 || floor(y) < 0 || ceil(y) > h-1)
            continue;
        x = int(round(x));
        y = int(round(y));
        z = vertices[nver*2 + i];

        if(z < depth_tmp[y*w + x])
            continue;
        if(fabs(z - depth_buffer[y*w + x]) < 1.5)
        {
            vis[i] = 1;
            depth_tmp[y*w + x] = z;         
        }
    }
}


void _get_triangle_buffer_core(
    int* triangle_buffer, double* vertices, int* triangles, 
    double* tri_depth, double* depth_buffer,
    int nver, int ntri,
    int h, int w, int c)
{
    int i;
    int x, y, z;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;

    for(i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[i];
        tri_p1_ind = triangles[ntri + i];
        tri_p2_ind = triangles[2*ntri + i];

        p0.x = vertices[tri_p0_ind]; p0.y = vertices[nver + tri_p0_ind];
        p1.x = vertices[tri_p1_ind]; p1.y = vertices[nver + tri_p1_ind];
        p2.x = vertices[tri_p2_ind]; p2.y = vertices[nver + tri_p2_ind];

        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
      
        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);


        if(x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for(y = y_min; y <= y_max; y++) //h
        {
            for(x = x_min; x <= x_max; x++) //w
            {
                p.x = x; p.y = y;
                if((tri_depth[i] > depth_buffer[y*w + x]) && isPointInTri(p, p0, p1, p2, h, w))
                {
                    depth_buffer[y*w + x] = tri_depth[i];
                    triangle_buffer[y*w + x] = i;
                }
            }
        }
    }
}


/* Judge whether the point is in the triangle
Method:
    http://blackpawn.com/texts/pointinpoly/
    https://blogs.msdn.microsoft.com/rezanour/2011/08/07/barycentric-coordinates-and-point-in-triangle-tests/
Args:
    point: [x, y] 
    tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
Returns:
    bool: true for in triangle
*/
bool isPointInTri(point p, point p0, point p1, point p2, int h, int w)
{   
    if(p.x < 2 || p.x > w - 3 || p.y < 2 || p.y > h - 3)
        return 1;
    // vectors
    point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    // dot products
    double dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    double dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    double dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    double dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    double dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    double inverDeno;
    if(dot00*dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1/(dot00*dot11 - dot01*dot01);

    double u = (dot11*dot02 - dot01*dot12)*inverDeno;
    double v = (dot00*dot12 - dot01*dot02)*inverDeno;

    // check if point in triangle
    return (u >= 0) && (v >= 0) && (u + v < 1);
}


void get_point_weight(double* weight, point p, point p0, point p1, point p2)
{   
    // vectors
    point v0, v1, v2;
    v0 = p2 - p0; 
    v1 = p1 - p0; 
    v2 = p - p0; 

    // dot products
    double dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    double dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    double dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    double dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    double dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    double inverDeno;
    if(dot00*dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1/(dot00*dot11 - dot01*dot01);

    double u = (dot11*dot02 - dot01*dot12)*inverDeno;
    double v = (dot00*dot12 - dot01*dot02)*inverDeno;

    // weight
    weight[0] = 1 - u - v;
    weight[1] = v;
    weight[2] = u;
}






void _get_correspondence_core(
    double* image, double* pncc_code, 
    double* uv,
    int nver,
    int h, int w, int c)
{
    int x, y, i, min_ind;
    double r, g, b, dr, dg, db, sum;
    double dis, min_dis;

    for(y = 0; y < h; y++)
    {
        for(x = 0; x < w; x++)
        {

            r = image[y*w*c + x*c];
            g = image[y*w*c + x*c + 1];
            b = image[y*w*c + x*c + 2];

            sum = r + g + b;
            if (sum < 0.07) 
                continue;

            min_dis = h + w;
            min_ind = 0;
            for(i = 0; i < nver; i++)
            {
                dr = r - pncc_code[i];
                dg = g - pncc_code[nver + i];
                db = b - pncc_code[2*nver + i];
                dis = dr*dr + dg*dg + db*db;
                if(dis < min_dis)
                { 
                    min_dis = dis;
                    min_ind = i;
                }
            }
            if(min_dis > 0.08)
                continue;
            uv[min_ind] = x;
            uv[nver + min_ind] = y;
            // printf("%d", min_ind);
        }
    }
}



// //// Possion Image Editing
// void possion_fusion(double*result)
// {
//     int x,y,n;





// }


/// obj write
// Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/core/Mesh.hpp
// void _write_obj(std::string filename,
//     double* vertices, double* uv_vertices, double* triangles,
//     int nver, int ntri)
// {
//     int i;

//     auto get_filename = [](const std::string& path) {
//         auto last_slash = path.find_last_of("/\\");
//         if (last_slash == std::string::npos)
//         {
//             return path;
//         }
//         return path.substr(last_slash + 1, path.size());
//     };

//     std::ofstream obj_file(filename);
//     std::string mtl_filename(filename);
//     // replace '.obj' at the end with '.mtl':
//     mtl_filename.replace(std::end(mtl_filename) - 4, std::end(mtl_filename), ".mtl");

//     obj_file << "mtllib " << get_filename(mtl_filename) << std::endl; // first line of the obj file
    
//     // write vertices 
//     for (i = 0; i < nver; ++i) 
//     {
//         obj_file << "v " << vertices[i] << " " << vertices[nver + i] << " " << vertices[2*nver + i] << " " << std::endl;
//     }

//     // write uv coordinates
//     for (i = 0; i < nver; ++i) 
//     {
//         obj_file << "vt " << uv_vertices[i] << " " << uv_vertices[nver + i] << std::endl;
//     }


//     // write triangles
//     for (i = 0; i < ntri; ++i) 
//     {
//         obj_file << "f " << triangles[i] + 1 << "/" << triangles[i] + 1 << " " << triangles[nver + i] + 1 << "/" << triangles[nver + i] + 1 << " " << triangles[2*nver + i] + 1 << "/" << triangles[2*nver + i] + 1 << std:endl;
//     }

//     //
//     std::ofstream mtl_file(mtl_filename);
//     std:string texture_filename(filename);
//     texture_filename.replace(std::end(texture_filename) - 4, std::end(texture_filename), ".isomap.png");

//     mtl_file << "newmtl FaceTexture" << std::endl;
//     mtl_file << "map_Kd " << get_filename(texture_filename) << std::endl;