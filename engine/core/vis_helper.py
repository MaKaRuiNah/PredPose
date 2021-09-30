#!/usr/bin/python
# -*- coding:utf8 -*-
import cv2
from random import random
import os.path as osp

from utils.utils_image import read_image, save_image
from datasets.process.keypoints_ord import coco2posetrack_ord_infer
from datasets.zoo.posetrack.pose_skeleton import PoseTrack_Official_Keypoint_Ordering, PoseTrack_Keypoint_Pairs
from utils.utils_color import COLOR_DICT
import matplotlib.pyplot as plt
import numpy as np

def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def draw_skeleton_in_origin_image(batch_image_list, batch_joints_list, batch_bbox_list, save_dir, vis_skeleton=True, vis_bbox=True):
    """
    :param batch_image_list:  batch image path
    :param batch_joints_list:   joints coordinates in image Coordinate reference system
    :batch_bbox_list: xyxy
    :param save_dir:
    :return: No return
    """

    skeleton_image_save_folder = osp.join(save_dir, "skeleton")
    bbox_image_save_folder = osp.join(save_dir, "bbox")
    together_save_folder = osp.join(save_dir, "SkeletonAndBbox")

    if vis_skeleton and vis_bbox:
        save_folder = together_save_folder
    else:
        save_folder = skeleton_image_save_folder
        if vis_bbox:
            save_folder = bbox_image_save_folder

    batch_final_coords = batch_joints_list

    for index, image_path in enumerate(batch_image_list):
        final_coords = batch_final_coords[index]
        final_coords = coco2posetrack_ord_infer(final_coords)
        bbox = batch_bbox_list[index]

        image_name = image_path[image_path.index("images") + len("images") + 1:]
        # image_name = image_path[image_path.index("frames") + len("frames") + 1:]

        vis_image_save_path = osp.join(save_folder, image_name)
        if osp.exists(vis_image_save_path):
            processed_image = read_image(vis_image_save_path)

            processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords, sure_threshold=0.2,
                                                                      flag_only_draw_sure=True) if vis_skeleton else processed_image
            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)
        else:
            image_data = read_image(image_path)
            processed_image = image_data.copy()

            processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords, sure_threshold=0.2,
                                                                      flag_only_draw_sure=True) if vis_skeleton else processed_image

            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)


def add_bbox_in_image(image, bbox):
    """
    :param image
    :param bbox   -  xyxy
    """

    color = (random() * 255, random() * 255, random() * 255)

    x1, y1, x2, y2 = map(int, bbox)
    image_with_bbox = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=6)
    return image_with_bbox


def add_poseTrack_joint_connection_to_image(img_demo, joints, sure_threshold=0.8, flag_only_draw_sure=False, ):
    for joint_pair in PoseTrack_Keypoint_Pairs:
        ind_1 = PoseTrack_Official_Keypoint_Ordering.index(joint_pair[0])
        ind_2 = PoseTrack_Official_Keypoint_Ordering.index(joint_pair[1])

        color = COLOR_DICT[joint_pair[2]]

        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]

        if x1 <= 5 and y1 <= 5: continue
        if x2 <= 5 and y2 <= 5: continue

        if flag_only_draw_sure is False:
            sure1 = sure2 = 1
        if sure1 > sure_threshold and sure2 > sure_threshold:
            # if sure1 > 0.8 and sure2 > 0.8:
            # cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=8)
            cv2.line(img_demo, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=6)
    return img_demo


def circle_vis_point(img, joints):
    for joint in joints:
        x, y, c = [int(i) for i in joint]
        cv2.circle(img, (x, y), 3, (255, 255, 255), thickness=3)

    return img
