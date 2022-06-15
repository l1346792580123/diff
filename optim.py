import os
from tqdm import tqdm
import numpy as np
import argparse
import trimesh
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
import nvdiffrast.torch as dr
from get_data import get_dtumvs
from utils import get_normals, get_ray_directions, get_rays, cokrender, gen_light_xyz, meshcleaning
from sap import DPSR, PSR2Mesh, grid_interp, sap_generate, gen_inputs


def main(idx):
    res = 128
    num_sample = 10000
    sig = 4
    batch = 8
    lr = 0.001
    h, w = 384, 512
    verts_weight = 30
    light_h = 4
    tex_resolution = 128
    mask_weight = 10
    mesh_name = 'space_carving_ret/dtu_%d.obj'%idx
    data_path = '' # set mvsdf_data path

    w2cs, projs, imgs, masks, verts_maps, depths, depth_cams, _, _ = get_dtumvs(data_path, idx, res=(512,384))

    resolution = (384, 512)
    num = imgs.shape[0]


    inputs, center, scale = gen_inputs(mesh_name, num_sample)
    
    inputs = inputs.cuda()
    inputs.requires_grad_(True)
    center = center.cuda()
    scale = scale.cuda()
    inputs_optimizer = Adam([{'params': inputs, 'lr': lr}])

    np_w2cs = w2cs.detach().cpu().numpy().transpose(0,2,1)
    np_projs = projs.detach().cpu().numpy().transpose(0,2,1)
    np_c2ws = []
    rays_origins = []
    rays_directions = []
    for i in range(num):
        np_projs[i,0,0] *= w / 2
        np_projs[i,1,1] *= h / 2
        np_projs[i,0,2] = (np_projs[i,0,2] + 1) * w / 2
        np_projs[i,1,2] = (np_projs[i,1,2] + 1) * h / 2

        np_focal = [np_projs[i,0,0], np_projs[i,1,1]]
        np_center = [np_projs[i,0,2], np_projs[i,1,2]]

        c2w = np.linalg.inv(np_w2cs[i])
        np_c2ws.append(c2w)

        direction = get_ray_directions(h, w, np_focal, np_center, False, False)
        rays_o, rays_d = get_rays(direction, torch.from_numpy(c2w[:3].astype(np.float32)))

        rays_origins.append(rays_o)
        rays_directions.append(rays_d)


    rays_origins = torch.stack(rays_origins, dim=0).cuda()
    rays_directions = torch.stack(rays_directions, dim=0).cuda()

    tex_grid = torch.ones(1, tex_resolution, tex_resolution, tex_resolution, 7).cuda() * 1e-2
    tex_grid.requires_grad_(True)

    lxyz, lareas = gen_light_xyz(light_h, 2*light_h, 10)
    lxyz = torch.from_numpy(lxyz.reshape(-1,3).astype(np.float32)).cuda()
    lareas = torch.from_numpy(lareas.reshape(1,-1,1).astype(np.float32)).cuda()
    light = torch.ones(num,light_h*2*light_h,3).cuda() * 0.1
    light.requires_grad_(True)

    optimizer = Adam([{'params': tex_grid, 'lr': 0.001}, {'params': light, 'lr': 0.001}])

    psr2mesh = PSR2Mesh.apply
    dpsr = DPSR((res,res,res), sig).cuda()

    glctx = dr.RasterizeGLContext()

    pbar = tqdm(range(301))

    for i in pbar:
        perm = torch.randperm(num).cuda()
        for k in range(0, num, batch):
            n = min(num, k+batch) - k
            w2c = w2cs[perm[k:k+batch]]
            proj = projs[perm[k:k+batch]]
            rays_direction = rays_directions[perm[k:k+batch]]
            img = imgs[perm[k:k+batch]]
            mask = masks[perm[k:k+batch]]
            verts_map = verts_maps[perm[k:k+batch]]

            vertices, faces, v, psr_grid, points = sap_generate(dpsr, psr2mesh, inputs, center, scale)
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            verts_normals = get_normals(vertsw[:,:,:3], faces.long())

            verts_tex = grid_interp(tex_grid.clamp(0), v.detach()).expand(n,-1,-1)
            
            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat = torch.cat([vertsw[:,:,:3], verts_normals, torch.ones_like(verts_normals), verts_tex], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, faces)
            rast_verts = feat[:,:,:,:3].contiguous()
            rast_normals = feat[:,:,:,3:6].contiguous()
            pred_mask = feat[:,:,:,6:9].contiguous()
            rast_tex = feat[:,:,:,9:16].contiguous()
            rast_verts = dr.antialias(rast_verts, rast_out, proj_verts, faces)
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces)


            valid_idx = torch.where((rast_out[:,:,:,3] > 0) & (mask[:,:,:,0] > 0) & (depths[perm[k:k+batch]] > 0))


            valid_verts = rast_verts[valid_idx] # N 3
            valid_normal = F.normalize(rast_normals[valid_idx], p=2, dim=1) # N 3
            valid_rays_d = rays_direction[valid_idx]
            valid_tex = rast_tex[valid_idx]


            color = cokrender(valid_verts.detach(), valid_normal.detach(), valid_tex, valid_idx, lxyz, light[perm[k:k+batch]].clamp(0), lareas, [n,h,w,3], valid_rays_d)
            color = dr.antialias(color, rast_out, proj_verts, faces)
            color_loss = 5 * F.l1_loss(color[valid_idx], img[valid_idx])

            mask_loss = mask_weight * F.mse_loss(pred_mask, mask)
            verts_loss = verts_weight * F.l1_loss(valid_verts, verts_map[valid_idx])

            total_loss = mask_loss + verts_loss + color_loss

            inputs_optimizer.zero_grad()
            optimizer.zero_grad()
            total_loss.backward()
            inputs_optimizer.step()
            optimizer.step()

            des = 'c:%.4f'%color_loss.item() + ' v:%.4f'%verts_loss.item() + ' m:%.4f'%mask_loss.item()
            pbar.set_description(des)

        if i % 50 == 0 and i != 0:

            if i == 150:
                num_sample = 60000

            with torch.no_grad():
                vertices, faces, v, psr_grid, points = sap_generate(dpsr, psr2mesh, inputs, center, scale)
                verts_tex = grid_interp(tex_grid, v)

                sample_verts = v.squeeze(0).detach().cpu().numpy()
                save_verts = vertices.squeeze(0).detach().cpu().numpy()
                np_faces = faces.squeeze(0).detach().cpu().long().numpy()
                save_mesh = trimesh.Trimesh(save_verts, np_faces, process=False, maintain_order=True)
                save_mesh.export('ret/%d.obj'%idx)
                meshcleaning('ret/%d.obj'%idx)

                torch.save(verts_tex.squeeze(), 'ret/%d_tex.pt'%idx)
                torch.save(light, 'ret/%d_light.pt'%idx)

                inputs, center, scale = gen_inputs('ret/%d.obj'%idx, num_sample)
                inputs = inputs.cuda()
                inputs.requires_grad_(True)
                center = center.cuda()
                scale = scale.cuda()

                
                del inputs_optimizer
                inputs_optimizer = Adam([{'params': inputs, 'lr': lr}])

                if i == 150:
                    batch = 4
                    res = 256
                    sig = 2
                    lr = 0.0005
                    del dpsr
                    dpsr = DPSR((res,res,res), sig).cuda()

                    new_tex_grid = F.interpolate(tex_grid.permute(0,4,1,2,3).contiguous(), size=(256,256,256), mode='trilinear').permute(0,2,3,4,1)
                    del tex_grid
                    tex_grid = new_tex_grid
                    tex_grid.requires_grad_(True)
                    optimizer = Adam([{'params': tex_grid, 'lr': 0.0005}, {'params': light, 'lr': 0.0005}])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=65)
    args = parser.parse_args()
    main(args.idx)