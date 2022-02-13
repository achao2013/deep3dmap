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
        