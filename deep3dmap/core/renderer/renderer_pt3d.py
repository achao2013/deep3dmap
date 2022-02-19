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
    def __init__(self, cfgs, image_size):
        self.texture_size=384
        self.raster_settings = RasterizationSettings(
            image_size=texture_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size = None,  # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin = None  # this setting is for coarse rasterization
        )
        self.lights = PointLights(device=device,ambient_color=((0, 0, 0),),diffuse_color=((1, 1, 1),),specular_color=((0, 0, 0),), location=[[0.0, 0.0, 10.0]])
        self.materials = Materials(device=device,ambient_color=((0, 0, 0),),diffuse_color=((1, 1, 1),),specular_color=((0, 0, 0),))

