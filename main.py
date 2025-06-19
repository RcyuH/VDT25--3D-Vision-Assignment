#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rcyuh
"""

import torch
import imageio
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader
)
from pytorch3d.renderer.cameras import look_at_view_transform
import matplotlib.pyplot as plt

def get_mesh_renderer(image_size=512, device="cpu"):
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )
    def renderer(meshes, cameras, lights):
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(meshes)
        shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
        images = shader(fragments, meshes)
        
        return images
    
    return renderer

def load_cow_mesh_vertex_color(device="cpu"):
    verts, faces, _ = load_obj("data/cow.obj")
    verts = verts.to(device)
    faces_idx = faces.verts_idx.to(device)
    red = torch.tensor([1.0, 0.0, 0.0], device=device)
    vert_colors = red.unsqueeze(0).repeat(verts.shape[0], 1)
    textures = TexturesVertex(verts_features=[vert_colors])
    meshes = Meshes(verts=[verts], faces=[faces_idx], textures=textures)
    
    return meshes


device = torch.device("cpu")
renderer = get_mesh_renderer(image_size=512, device=device)
meshes = load_cow_mesh_vertex_color(device=device)

images = []
num_views = 60
for azim in torch.linspace(0, 360.0, steps=num_views):
    R, T = look_at_view_transform(dist=3.0, elev=0.0, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, fov=60.0, device=device)
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    image = rend[0, ..., :3].cpu().numpy()
    images.append((image * 255).astype("uint8"))

imageio.mimsave("images/cow_turntable.gif", images, fps=15)
    

verts, faces, _ = load_obj("data/cow.obj")
verts, faces = verts.to(device), faces.verts_idx.to(device)
light_blue = torch.tensor([0.5,0.7,1.0], device=device)
vert_colors = light_blue.unsqueeze(0).repeat(verts.shape[0],1)
meshes = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(verts_features=[vert_colors]))
raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)

def renderer(meshes, cameras, lights):
    frags = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)(meshes)
    
    return HardPhongShader(device=device, cameras=cameras, lights=lights)(frags, meshes)

R0 = torch.eye(3, device=device).unsqueeze(0)
T0 = torch.tensor([[0.0,0.0,3.0]], device=device)

Rnew, Tnew = look_at_view_transform(dist=3.0, elev=90.0, azim=0.0, device=device)
Rrel = Rnew  
Trel = Tnew - (Rrel @ T0.unsqueeze(-1)).squeeze(-1)

R = Rrel @ R0
T = (Rrel @ T0.unsqueeze(-1)).squeeze(-1) + Trel

cams = FoVPerspectiveCameras(R=R, T=T, fov=60.0, device=device)
lights = PointLights(location=[[0.0,5.0,0.0]], device=device)
img = renderer(meshes, cams, lights)[0,...,:3].cpu().numpy()
plt.imsave("images/cow_above.jpg", (img*255).astype("uint8"))