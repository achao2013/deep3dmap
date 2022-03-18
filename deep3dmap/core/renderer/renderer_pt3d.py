import torch
import numpy as np
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles,Rotate
# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds, Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    OpenGLOrthographicCameras,
    #SfMPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    PointsRasterizationSettings,
    MeshRenderer,
    PointsRenderer,
    MeshRasterizer,
    PointsRasterizer,
    TexturedSoftPhongShader,
    NormWeightedCompositor,
    BlendParams,
    AlphaCompositor
    )

class Pt3dRenderer():
    def __init__(self, device, texture_size, lookview):
        
        self.raster_settings = RasterizationSettings(
            image_size=texture_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size = None,  # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin = None  # this setting is for coarse rasterization
        )
        self.lights = PointLights(device=device,ambient_color=((0, 0, 0),),diffuse_color=((1, 1, 1),),specular_color=((0, 0, 0),), location=[[0.0, 0.0, 10.0]])
        self.materials = Materials(device=device,ambient_color=((0, 0, 0),),diffuse_color=((1, 1, 1),),specular_color=((0, 0, 0),))
        self.lookview=lookview.view(1,3)
        self.device=device
    def sample(self,normals,angles,triangles,imgs,template_uvs3d,face_project):
        #rot=Rotate(R, device=device)
        #normals_transformed = rot.transform_normals(normals.repeat(batchsize,1,1))
        batchsize=angles.shape[0]
        vertexsize=normals.shape[1]
        trisize=triangles.shape[1]
        RR = euler_angles_to_matrix(angles, "XYZ")
        rot=Rotate(RR)
        normals_transformed = rot.transform_normals(normals)
        coefs = torch.sum(torch.mul(normals_transformed, self.lookview.repeat(batchsize,vertexsize,1)), 2)
        ver_visibility = torch.ones(batchsize,vertexsize).cuda()
        ver_visibility[coefs < 0] = 0

        visible_veridx = (ver_visibility>0).nonzero()
        tri_visibility = (triangles.reshape(batchsize,trisize,3,1) == visible_veridx.reshape(batchsize,1,1,vertexsize)).any(-1)
        visible_triidx = (torch.sum(tri_visibility, 2)>0).nonzero()
        used_faces = triangles[visible_triidx]
        tex = Textures(verts_uvs=face_project, faces_uvs=used_faces, maps=imgs.permute(2,3,1))
        mesh = Meshes(
            verts=[template_uvs3d], faces=[used_faces], textures=tex)
        R_, T_ = look_at_view_transform(2.7, torch.zeros(batchsize).cuda(), torch.zeros(batchsize).cuda())
        camera = OpenGLOrthographicCameras(device=self.device, R=R_, T=T_)
        #camera = OpenGLOrthographicCameras(R=R_, T=T_)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=self.raster_settings
                ),
            shader=TexturedSoftPhongShader(
                device=self.device,
                cameras=camera,
                blend_params=BlendParams(background_color=(0,0,0))
                )
        )
        uv_images = renderer(mesh)
        mask = Textures(verts_uvs=face_project, faces_uvs=used_faces, maps=torch.ones_like(imgs.permute(2,3,1)))
        mesh_mask = Meshes(
            verts=[template_uvs3d], faces=[used_faces], textures=mask)
        uv_mask = renderer(mesh_mask)
        return uv_images,uv_mask
        
