import numpy as np
import cv2
from glob import glob
import os
from os.path import join
from tqdm import tqdm
import trimesh
from utils import convert_sdf_to_ply, load_K_Rt_from_P
import argparse

def main(data_path, idx):

    silhouette = []
    projs = []
    w2cs = []

    # for dtu
    camera_dict = np.load(join('%s/scan%d/imfunc4/cameras_hd.npz'%(data_path, idx)))

    num = 49

    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(num)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(num)]

    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        proj, w2c = load_K_Rt_from_P(P[:3, :4])

        projs.append(proj)
        w2cs.append(w2c)

    for i in range(num):
        img = cv2.imread('%s/scan%d/imfunc4/pmask/%03d.png'%(data_path, idx,i))
        mask = img[:,:,0] > 0

        silhouette.append(mask)

    N = 128
    imgH, imgW = 1200, 1600
    voxel_origin = [-1.1, -1.1, -1.1]
    x_size = 2.2 / (N - 1)
    y_size = 2.2 / (N - 1)
    z_size = 2.2 / (N - 1)

    overall_index = np.arange(0, N ** 3, 1, dtype=np.int64)
    pts = np.zeros([N ** 3, 3], dtype=np.float32)

    pts[:, 2] = overall_index % N
    pts[:, 1] = (overall_index // N) % N
    pts[:, 0] = ((overall_index // N) // N) % N

    pts[:, 0] = (pts[:, 0] * x_size) + voxel_origin[0]
    pts[:, 1] = (pts[:, 1] * y_size) + voxel_origin[1]
    pts[:, 2] = (pts[:, 2] * z_size) + voxel_origin[2]

    pts = np.vstack((pts.T, np.ones((1, N**3))))

    filled = []

    for calib, transform, im in tqdm(zip(w2cs, projs, silhouette)):
        uvs = transform @ calib @ pts
        uvs[0] = uvs[0] / uvs[2]
        uvs[1] = uvs[1] / uvs[2]
        uvs = np.round(uvs).astype(np.int32)
        x_good = np.logical_and(uvs[0] >= 0, uvs[0] < imgW)
        y_good = np.logical_and(uvs[1] >= 0, uvs[1] < imgH)
        good = np.logical_and(x_good, y_good)
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        res = im[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res 
        
        filled.append(fill)

    filled = np.vstack(filled)

    occupancy = -np.sum(filled, axis=0)

    occupancy = occupancy.reshape(N,N,N)

    convert_sdf_to_ply(occupancy, voxel_origin, [x_size, y_size, z_size], '1.ply', level=-(num-4))
    mesh = trimesh.load('1.ply', process=False, maintain_order=True)
    mesh.export('space_carving_ret/dtu_%d.obj'%idx)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--idx', type=int, default=65)
    args = parser.parse_args()
    main(args.data_path, args.idx)