import numpy as np
import math
import cv2
import trimesh
import re
import plyfile
import skimage.measure
import torch
import torch.nn as nn
import torch.nn.functional as F


def meshcleaning(file_name):

    mesh = trimesh.load(file_name)
    cc = mesh.split(only_watertight=False)    

    out_mesh = cc[0]
    bbox = out_mesh.bounds
    area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
    for c in cc:
        bbox = c.bounds
        if area < (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1]):
            area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
            out_mesh = c
    
    out_mesh.export(file_name)


def get_normals(vertices, faces):
    '''
    vertices b n 3
    faces f 3
    '''
    verts_normals = torch.zeros_like(vertices)

    vertices_faces = vertices[:, faces] # b f 3 3

    verts_normals.index_add_(
        1,
        faces[:, 1],
        torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 2],
        torch.cross(
            vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
            vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 0],
        torch.cross(
            vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
            vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
            dim=2,
        ),
    )

    verts_normals = F.normalize(verts_normals, p=2, dim=2)

    return verts_normals

def SmithG1(alpha, h, m, n, eps=1e-7):
    if len(m.shape) == 2:
        cos_theta_v = torch.einsum('ij,ij->i', n, m)[:, None] # N 1
        cos_theta_g = torch.einsum('ijk,ik->ij', h, m) # N L
    else:
        cos_theta_v = torch.einsum('ik,ijk->ij', n, m) # N L
        cos_theta_g = torch.einsum('ijk,ijk->ij', h, m) # N L
    div_g = cos_theta_g / (cos_theta_v + eps)
    chi_g = torch.where(div_g > 0, 1., 0.)
    cos_theta_v_sq = torch.square(cos_theta_v).clamp(0,1)
    tan_theta_v_sq = (1 - cos_theta_v_sq) / (cos_theta_v_sq + eps).clamp(0)
    denom_g = 1 + torch.sqrt(1 + alpha ** 2 * tan_theta_v_sq)
    g = (chi_g * 2) / (denom_g + eps) # N L

    return g

def cokrender(valid_verts, valid_normal, valid_tex, valid_idx, lxyz, light, lareas, size, rays_d=None, f0=0.04, eps=1e-7):
    albedo = valid_tex[:,:3].clamp(0)
    specular = valid_tex[:,3:6].clamp(0)
    rough = valid_tex[:,6:].clamp(0)
    alpha = rough**2
    surf2light = F.normalize(lxyz.unsqueeze(0) - valid_verts.detach().unsqueeze(1), p=2, dim=2) # N L 3
    if rays_d is None:
        surf2camera = F.normalize(-valid_verts.detach(), p=2, dim=1) # N 3
    else:
        surf2camera = -rays_d

    h = F.normalize(surf2light + surf2camera.unsqueeze(1), p=2, dim=2) # N L 3

    # Fresnel
    cos_theta = torch.einsum('ijk,ijk->ij', surf2light, h)
    f = f0 + (1 - f0) * (1 - cos_theta) ** 5

    # D
    cos_theta_d = torch.einsum('ijk,ik->ij', h, valid_normal)
    chi_d = torch.where(cos_theta_d > 0, 1, 0)
    cos_theta_m_sq = torch.square(cos_theta_d)
    tan_theta_m_sq = (1 - cos_theta_m_sq) / (cos_theta_m_sq+eps)
    denom_d = np.pi * torch.square(cos_theta_m_sq) * torch.square(alpha ** 2 + tan_theta_m_sq)
    d = (alpha ** 2 * chi_d) / (denom_d+eps) # N L

    # GGX
    g = SmithG1(alpha, h, surf2camera, valid_normal, eps) * SmithG1(alpha, h, surf2light, valid_normal, eps)

    l_dot_n = torch.einsum('ijk,ik->ij', surf2light, valid_normal) # N L
    v_dot_n = torch.einsum('ij,ij->i', surf2camera, valid_normal)
    denom = 4 * torch.abs(l_dot_n) * torch.abs(v_dot_n)[:, None]

    microfacet = (f * g * d) / (denom + eps)# N L

    glossy = specular.unsqueeze(1) * microfacet.unsqueeze(2)

    # TODO energy conservation ks kd
    brdf = (albedo / np.pi).unsqueeze(1) + glossy # N L 3

    light_vis = (l_dot_n > 0).float()

    light_flat = light_vis.unsqueeze(2) * light[valid_idx[0]] # N L 3
    light_pix_contrib = brdf * light_flat * l_dot_n[:, :, None] * lareas # NxLx3
    rgb = torch.sum(light_pix_contrib, axis=1).clamp(0,1) # Nx3
    rgb = linear2srgb(rgb)

    color = torch.zeros(size).to(valid_verts.device)
    color[valid_idx] = rgb

    return color


def convert_sdf_to_ply(sdf_values, voxel_origin, voxel_size, file_name, level=0.):
    if isinstance(sdf_values, torch.Tensor):
        sdf_values = sdf_values.detach().cpu().numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(sdf_values, 
                                level=level, spacing=voxel_size)

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    normals_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])
        normals_tuple[i] = tuple(normals[i,:])

    faces_building = []

    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))

    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
    ply_data = plyfile.PlyData([el_verts, el_faces])
    # el_normals = plyfile.PlyElement.describe(normals_tuple, "normal")
    # ply_data = plyfile.PlyData([el_verts, el_faces, el_normals])
    ply_data.write(file_name)

def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    # c2w
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    # convert to w2c
    pose = np.linalg.inv(pose)

    return intrinsics, pose


# code adapted from https://github.com/kwea123/nerf_pl/blob/master/datasets/ray_utils.py

def create_meshgrid(height, width, normalized_coordinates=False):
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, dtype=torch.float)

    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW

    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2

def get_ray_directions(H, W, focal, c=None, minus_z=True, minus_y=True):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    i, j = grid.unbind(-1)

    if isinstance(focal, float):
        fx = focal
        fy = focal
    elif isinstance(focal, list):
        fx, fy = focal

    if minus_z:
        z = -torch.ones_like(i)
    else:
        z = torch.ones_like(i)

    if c is None:
        c = [W/2, H/2]

    x = (i-c[0])/fx
    if minus_y:
        y = -(j-c[1])/fy
    else:
        y = (j-c[1])/fy

    directions = torch.stack([x,y,z], -1)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    # rays_d = rays_d.view(-1, 3)
    # rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

# code adapted from https://github.com/google/nerfactor

def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart

def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas

def linear2srgb(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    else:
        pow_func = np.power
        where_func = np.where

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (pow_func(tensor_0to1+1e-5, 1 / srgb_exponent)) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


# code adapted from mvsdf https://github.com/jzhangbs/MVSDF


def load_pfm(file: str):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(br'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = data[::-1, ...]  # cv2.flip(data, 0)
    return data


def load_cam(file: str, max_d, interval_scale=1, override=False):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    with open(file) as f:
        words = f.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 31:
        if override:
            cam[1][3][0] = words[27]
            cam[1][3][1] = (float(words[30]) - float(words[27])) / (max_d - 1)
            cam[1][3][2] = max_d
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam