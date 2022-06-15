import os
from os.path import join
from glob import glob
import json
import numpy as np
import cv2
import imageio
import trimesh
import torch
import torch.nn.functional as F
from utils import load_K_Rt_from_P, load_pfm, load_cam


def get_dtumvs(data_path, scan_id, res=(800,600), use_mask=True):
    camera_dict = np.load(join(data_path, 'scan%d/imfunc4/cameras_hd.npz'%scan_id))
    num = 49

    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(num)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(num)]

    size = (scale_mats[0][0,0] * 2).astype(np.float32)
    center = (scale_mats[0][:3,3]).astype(np.float32)

    projs = []
    w2cs = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        proj, w2c = load_K_Rt_from_P(P)

        proj[0,0] = proj[0,0] / 800.
        proj[0,2] = proj[0,2] / 800. - 1.
        proj[1,1] = proj[1,1] / 600.
        proj[1,2] = proj[1,2] / 600. - 1.
        proj[2,2] = 0.
        proj[2,3] = -1.
        proj[3,2] = 1.0
        proj[3,3] = 0.0

        proj = torch.from_numpy(proj.astype(np.float32)).cuda()
        w2c = torch.from_numpy(w2c.astype(np.float32)).cuda()

        projs.append(proj)
        w2cs.append(w2c)

    # transpose for right multiplication
    w2cs = torch.stack(w2cs, dim=0).permute(0,2,1).contiguous()
    projs = torch.stack(projs, dim=0).permute(0,2,1).contiguous()

    imgs = []
    masks = []
    depths = []
    depth_cams = []
    verts_maps = []
    x_coord = (np.arange(512) + 0.5)[None].repeat(384,0)
    y_coord = (np.arange(384) + 0.5)[None].repeat(512, 0).transpose()
    ones = np.ones_like(x_coord)
    indices_grid = np.stack([x_coord, y_coord, ones], axis=-1) # h w 3

    for i in range(num):
        img = cv2.imread(join(data_path, 'scan%d/imfunc4/image_hd/%06d.png'%(scan_id, i)))
        depth = np.ascontiguousarray(load_pfm(join(data_path, 'scan%d/imfunc4/depth/%03d.pfm'%(scan_id, i)))).astype(np.float32)
        mask = cv2.imread(join(data_path, 'scan%d/imfunc4/pmask/%03d.png'%(scan_id,i)))
        if use_mask:
            dep_mask = cv2.resize(mask, (512,384), interpolation=cv2.INTER_NEAREST)
            depth[dep_mask[:,:,0]==0] = 0

        depth_cam = load_cam(join(data_path, 'scan%d/cam_%08d_flow3.txt'%(scan_id, i)), 256, 1).astype(np.float32)

        inv_cam2 = np.linalg.inv(depth_cam[1,:3,:3])
        idx_cam = np.einsum('lk,ijk->ijl', inv_cam2, indices_grid) # h w 3
        idx_cam = idx_cam / (idx_cam[...,-1:]+1e-9) * depth[:,:,None]
        idx_cam_homo = np.concatenate([idx_cam, np.ones_like(idx_cam[...,-1:])], axis=-1)

        inv_cam = np.linalg.inv(depth_cam[0])
        idx_world_homo = np.einsum('lk,ijk->ijl', inv_cam, idx_cam_homo) # n h w 4
        idx_world_homo = (idx_world_homo / (idx_world_homo[...,-1:]+1e-9))[:,:,:3]  # h w 4
        verts_map = (idx_world_homo - center) / size * 2
        verts_maps.append(torch.from_numpy(verts_map.astype(np.float32)).cuda())

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask[mask>0] = 1

        img = cv2.resize(img, res)
        mask = cv2.resize(mask, res, interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy((img/255.)).float().cuda()
        mask  = torch.from_numpy(mask).float().cuda()
        depth = torch.from_numpy(depth).cuda()
        depth_cam = torch.from_numpy(depth_cam).cuda()

        imgs.append(img)
        masks.append(mask)
        depths.append(depth)
        depth_cams.append(depth_cam)

    masks = torch.stack(masks, dim=0)
    imgs = torch.stack(imgs, dim=0)
    depths = torch.stack(depths, dim=0)
    depth_cams = torch.stack(depth_cams, dim=0)
    verts_maps = torch.stack(verts_maps, dim=0)
    center = torch.from_numpy(center).cuda()

    return w2cs, projs, imgs, masks, verts_maps, depths, depth_cams, center, size