import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    PerspectiveCameras,
    BlendParams,
    SoftSilhouetteShader,
    SoftPhongShader,
    look_at_view_transform,
)
from typing import Union, List, Tuple, Optional

def rasterize_meshes(
    meshes: Meshes,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    faces_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_faces_per_bin: Optional[int] = None,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = False,
    cull_backfaces: bool = False,
    z_clip_value: Optional[float] = None,
    cull_to_frustum: bool = False,
):
    # Ensure image_size is a tuple
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    # Define rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
        perspective_correct=perspective_correct,
        clip_barycentric_coords=clip_barycentric_coords,
        cull_backfaces=cull_backfaces,
        z_clip_value=z_clip_value,
        cull_to_frustum=cull_to_frustum,
    )
    
    # Create a camera (assuming a simple perspective camera)
    R, T = look_at_view_transform(2.7, 0, 0)  