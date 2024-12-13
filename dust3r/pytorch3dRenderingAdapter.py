# 241104 예원 작성 코드


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import matplotlib.pyplot as plt
import os
import copy
import cv2


def Rendering_Adapter(scene, cam_num_list, pts_num_list):

    focals = scene.get_focals()
    cams2world = scene.get_im_poses()
    imsizes = scene.imsizes
    images = scene.imgs
    pts3d = scene.get_pts3d()

    # 포인트 클라우드 시각화
    ##############################
    all_points = []
    all_colors = []

    for idx in pts_num_list:
        points = pts3d[idx].view(-1, 3).to('cuda')
        colors = images[idx].reshape(-1, 3)
        all_points.append(points)
        all_colors.append(colors)

    merged_points = torch.cat(all_points, dim=0)  # .requires_grad_(True)
    merged_colors = np.concatenate(all_colors, axis=0)
    merged_colors = torch.from_numpy(merged_colors).to('cuda')

    point_cloud = Pointclouds(points=[merged_points], features=[merged_colors])  ###추가
    ##############################

    # 카메라 포즈 변환 및 렌더링
    ##############################
    check = 0
    for i in cam_num_list:
        if i == 999:
            i = 0
            check = 1
        image_width = imsizes[i][0]  # 512
        image_height = imsizes[i][1]  # 336
        focal = focals[i]

        ###
        if check == 1:
            i = 999
        ###
        if i != 999:
            pose_c2w = cams2world[i]
        else:
            R0, t0 = cams2world[0][:3, :3], cams2world[0][:3, 3]
            R1, t1 = cams2world[1][:3, :3], cams2world[1][:3, 3]
            # Step 2: 중간 병진 벡터 계산
            t_mid = (t0 + t1) / 2.0

            # Step 3: 중간 회전 행렬 계산 (Slerp 또는 평균)
            # 여기서는 간단히 두 회전 행렬을 평균한 후 직교화합니다.
            R_mid = (R0 + R1) / 2.0

            # 직교화 (Gram-Schmidt 과정) - 회전 행렬을 다시 올바른 형태로 만듭니다.
            u, _, v = torch.svd(R_mid)  # 특이값 분해 (SVD)
            R_mid_orthogonal = torch.matmul(u, v.t())

            # Step 4: 중간 Extrinsic Matrix 생성
            cams2world_mid = torch.eye(4, device=cams2world[0].device)
            cams2world_mid[:3, :3] = R_mid_orthogonal
            cams2world_mid[:3, 3] = t_mid

            print("Middle Camera Extrinsic Matrix:")
            print(cams2world_mid)
            pose_c2w = cams2world_mid

        pose_w2c = torch.inverse(pose_c2w)
        ####
        # pose_w2c에서 회전 행렬 추출
        initial_rotation_matrix = pose_w2c[:3, :3]  # 3x3 회전 행렬

        # 초기 회전 행렬을 쿼터니언으로 변환
        initial_quaternion = matrix_to_quaternion(initial_rotation_matrix)

        # 쿼터니언을 학습 가능한 파라미터로 설정 (초기값으로 pose_w2c의 회전 행렬 사용)
        quaternion = torch.nn.Parameter(initial_quaternion)  # 학습 가능한 쿼터니언

        # 쿼터니언을 회전 행렬로 변환
        R_opencv = quaternion_to_matrix(F.normalize(quaternion, dim=0).unsqueeze(0))  # Batch dimension 추가
        tvec_opencv = pose_w2c[:3, 3].unsqueeze(0)  # Translation vector with batch dimension

        # Camera calibration matrix (intrinsics) in OpenCV format
        camera_matrix = torch.tensor([
            [focal, 0, image_width / 2.0],
            [0, focal, image_height / 2.0],
            [0, 0, 1]
        ], device='cuda').unsqueeze(0)  # Add batch dimension

        # Image size tensor with batch dimension
        image_size = torch.tensor([[image_height, image_width]], device='cuda')

        # Convert OpenCV camera parameters to PyTorch3D PerspectiveCameras
        # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/utils/camera_conversions.html
        cameras = cameras_from_opencv_projection(
            R=R_opencv,
            tvec=tvec_opencv,
            camera_matrix=camera_matrix,
            image_size=image_size
        )

        raster_settings = PointsRasterizationSettings(
            image_size=(image_height, image_width),
            radius=1.0 / image_height * 3.0,  # 2.0,
            # points_per_pixel=10,
            # bin_size=0  # Naive rasterization 사용
        )

        # 렌더러 구성
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        rendered_image = renderer(point_cloud)[0]



        # 이미지 값을 0에서 1 사이로 정규화하거나 클램핑
        rendered_image_clamped = torch.clamp(rendered_image, min=0.0, max=1.0)
        # 이미지 저장
        plt.imsave(f'/home/asc/PycharmProjects/SuperIntelligenc_Project/dust3r/outputs/camera{i}_pts{pts_num_list}.png',
                   rendered_image_clamped.detach().cpu().numpy())


    return


