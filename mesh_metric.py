
import torch
from  submodules.chamfer_distance import ChamferDistance
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    return positions

# 初始化 ChamferDistance 对象
chamfer_dist = ChamferDistance()
gt_ply='/root/autodl-tmp/data/scan6/mesh.ply'
test_ply=['/root/autodl-tmp/mesh/sacn6_sence_1/scan6_2_relative_depth.ply','/root/autodl-tmp/mesh/sacn6_sence_1/scan6_2_1999.ply','/root/autodl-tmp/mesh/sacn6_sence_1/scan6_1_depth.ply','/root/autodl-tmp/mesh/sacn6_sence_1/scan6_base.ply']

gt_pcd = fetchPly(gt_ply)


target_cloud = torch.tensor(np.asarray(gt_pcd)).float().unsqueeze(0).cuda()
for i in test_ply:
    test_pcd= fetchPly(i)
    source_cloud = torch.tensor(np.asarray(test_pcd)).float().unsqueeze(0).cuda()
    dist1,dist2 = chamfer_dist(source_cloud, target_cloud)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    #cham,_=chamfer_distance(source_cloud, target_cloud)
    print(loss)    
# 计算 Chamfer 距离


