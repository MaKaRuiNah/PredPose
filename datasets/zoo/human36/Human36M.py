#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import os.path as osp
import torch
import copy
import random
import cv2
import json
from pycocotools.coco import COCO
import logging
from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored

from engine.core.vis_helper import vis_keypoints

from datasets.zoo.posetrack.posetrack_utils import video2filenames, evaluate_simple
from utils.utils_json import read_json_from_file, write_json_to_file
from utils.utils_bbox import box2cs
from utils.utils_image import read_image
from utils.utils_folder import create_folder
from utils.utils_registry import DATASET_REGISTRY
from utils.pose_utils import world2cam, cam2pixel, pixel2cam, rigid_align, process_bbox

from datasets.process import get_affine_transform, fliplr_joints, exec_affine_transform, generate_heatmaps, half_body_transform, \
    convert_data_to_annorect_struct

from datasets.transforms import build_transforms
from datasets.zoo.base import VideoDataset

from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE


@DATASET_REGISTRY.register()
class Human36M(VideoDataset):
    """
        PoseTrack
    """

    def __init__(self, cfg, phase, **kwargs):
        super(Human36M, self).__init__(cfg, phase, **kwargs)

        self.img_dir = '/mancheng/data/Human36M/images'
        self.annot_dir = '/mancheng/data/Human36M/annotations'
        self.human_bbox_root_dir = '/mancheng/data/Human36M/bbox_root/bbox_root_human36m_output.json'

        self.joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))

        self.joints_weight = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                      dtype=np.float32).reshape((self.num_joints, 1))
        self.joints_have_depth = True
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)  # exclude Thorax

        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']

        self.root_idx = self.joints_name.index('Pelvis')
        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')

        self.transform = build_transforms(cfg, phase)
        self.random_aux_frame = cfg.DATASET.RANDOM_AUX_FRAME
        self.bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        self.sigma = cfg.MODEL.SIGMA

        if self.phase != TRAIN_PHASE:
            temp_subCfgNode = cfg.VAL if self.phase == VAL_PHASE else cfg.TEST
            self.nms_thre = temp_subCfgNode.NMS_THRE
            self.image_thre = temp_subCfgNode.IMAGE_THRE
            self.soft_nms = temp_subCfgNode.SOFT_NMS
            self.oks_thre = temp_subCfgNode.OKS_THRE
            self.in_vis_thre = temp_subCfgNode.IN_VIS_THRE
            #self.bbox_file = temp_subCfgNode.COCO_BBOX_FILE
            self.use_gt_bbox = temp_subCfgNode.USE_GT_BBOX

        self.protocol = 1

        self.data = self._list_data()
        #self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        #self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.model_input_type = cfg.DATASET.INPUT_TYPE

        self.show_data_parameters()
        self.show_samples()

    def __getitem__(self, item_index):
        #item_index = min(len(self.data),item_index+1)
        data_item = copy.deepcopy(self.data[item_index])
        if self.model_input_type == 'single_frame':
            return self._get_single_frame(data_item)
        elif self.model_input_type == 'spatiotemporal_window':
            return self._get_spatiotemporal_window(data_item)

    def get_subsampling_ratio(self):
        if self.phase == "train":
            return 5
        elif self.phase == "test" or self.phase == "validate":
            return 2
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.phase == "train":
            if self.protocol == 1:
                subject = [1, 5, 6, 7, 8, 9]
            elif self.protocol == 2:
                subject = [1, 5, 6, 7, 8]
        elif self.phase == "test" or self.phase == "validate":
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9, 11]
        else:
            assert 0, print("Unknown subset")

        return subject

    def _get_spatiotemporal_window(self, data_item):
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = read_image(image_file_path)

        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        if zero_fill == 6:
            is_posetrack18 = True
        else:
            is_posetrack18 = False

        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))


        prev_idx1 = max(1,current_idx -1)
        prev_idx2 = max(1,current_idx -2)

        prev1_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx1).zfill(zero_fill) + '.jpg')
        prev2_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx2).zfill(zero_fill) + '.jpg')

        # checking for files existence
        if not osp.exists(prev1_image_file):
            error_msg = "Can not find image :{}".format(prev1_image_file)
            self.logger.error(error_msg)
            raise Exception(error_msg)
        if not osp.exists(prev2_image_file):
            error_msg = "Can not find image :{}".format(prev2_image_file)
            self.logger.error(error_msg)
            raise Exception(error_msg)

        data_numpy_prev1 = read_image(prev1_image_file)
        data_numpy_prev2 = read_image(prev2_image_file)

        if self.color_rgb:
            # cv2 read_image  color channel is BGR
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy_prev1 = cv2.cvtColor(data_numpy_prev1, cv2.COLOR_BGR2RGB)
            data_numpy_prev2 = cv2.cvtColor(data_numpy_prev2, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            self.logger.error('=> fail to read {}'.format(image_file_path))
            raise ValueError('Fail to read {}'.format(image_file_path))
        if data_numpy_prev1 is None:
            self.logger.error('=> PREV SUP: fail to read {}'.format(prev1_image_file))
            raise ValueError('PREV SUP: Fail to read {}'.format(prev1_image_file))
        if data_numpy_prev2 is None:
            self.logger.error('=> NEXT SUP: fail to read {}'.format(prev2_image_file))
            raise ValueError('NEXT SUP: Fail to read {}'.format(prev2_image_file))

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids, self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                data_numpy_prev1 = data_numpy_prev1[:, ::-1, :]
                data_numpy_prev2 = data_numpy_prev2[:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_prev1 = cv2.warpAffine(data_numpy_prev1, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_prev2 = cv2.warpAffine(data_numpy_prev2, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            input_prev1 = self.transform(input_prev1)
            input_prev2 = self.transform(input_prev2)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]
        # target_heatmaps, target_heatmaps_weight = self._generate_target(joints, joints_vis, self.heatmap_size, self.num_joints)

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        meta = {
            'image': image_file_path,
            'prev_sup_image': prev1_image_file,
            'next_sup_image': prev2_image_file,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
        }

        #return input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, meta
        return input_x, input_prev1, input_prev2, target_heatmaps, target_heatmaps_weight, meta

    def _get_single_frame(self, data_item):
        img_path = data_item['img_path']
        img_id = data_item['img_id']

        center =  data_item['center']
        scale = data_item['scale']
        bbox = data_item['bbox']
        joints = data_item['joints']
        joints_vis = data_item['joints_vis']
        frame_id = data_item['frame_id']
        action_id = data_item['action_id']
        subaction_id = data_item['subaction_id']
        cam_id = data_item['cam_id']


        data_numpy = read_image(img_path)

        if self.color_rgb:
            # cv2 read_image  color channel is BGR
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        if data_numpy is None:
            self.logger.error('=> fail to read {}'.format(img_path))
            raise ValueError('Fail to read {}'.format(img_path))

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)

        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                 flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]
        # target_heatmaps, target_heatmaps_weight = self._generate_target(joints, joints_vis, self.heatmap_size, self.num_joints)
        #joints[:,0] = joints/self.image_size[0]


        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size,
                                                                    self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        meta =  {
            'img_path': img_path,
            'img_id': img_id,
            'center': center,
            'scale': scale,
            'bbox': bbox,
            'joints': joints,  # [org_img_x, org_img_y, depth - root_depth]
            'joints_vis': joints_vis,
            'frame_id': frame_id,
            'action_id': action_id,
            'subaction_id': subaction_id,
            'cam_id': cam_id,
            'rotation': r,
            'score': score,
        }

        # return input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, meta
        return input_x, target_heatmaps, target_heatmaps_weight, meta

    def _list_data(self):
        if self.is_train or self.use_gt_bbox:
            # use bbox from annotation
            data = self._load_coco_keypoints_annotations()
        else:
            # use bbox from detection
            data = self._load_detection_results()
        return data

    def _load_coco_keypoints_annotations(self):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
        """

        print('Load data of H36M Protocol ' + str(self.protocol))

        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_dir, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k, v in annot.items():
                    db.dataset[k] = v
            else:
                for k, v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_dir, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_dir, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                joints[str(subject)] = json.load(f)
        db.createIndex()

        if self.phase == TEST_PHASE and not self.use_gt_bbox:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']),
                                                               'root': np.array(annot[i]['root_cam'])}
        else:
            print("Get bounding box and root from groundtruth")

        data = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']

            # check subject and frame_idx
            subject = img['subject']
            frame_idx = img['frame_idx']
            if subject not in subject_list:
                continue
            if frame_idx % sampling_ratio != 0:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'],
                                                                              dtype=np.float32), np.array(
                cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)

            # project world coordinate to cam, image coordinate space
            action_idx = img['action_idx']
            subaction_idx = img['subaction_idx']
            frame_idx = img['frame_idx']
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)],
                                   dtype=np.float32)
            #joint_world = self.add_thorax(joint_world)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
            #joint_img[:, 2] = joint_img[:, 2] - joint_cam[self.root_idx, 2]
            joint_img[:, 2] = 0
            joint_vis = np.ones((self.num_joints, 3))
            joint_vis[:,2] = 0

            # if self.data_split == 'test' and not cfg.use_gt_info:
            #     bbox = bbox_root_result[str(image_id)][
            #         'bbox']  # bbox should be aspect ratio preserved-extended. It is done in RootNet.
            #     root_cam = bbox_root_result[str(image_id)]['root']
            # else:

            bbox = process_bbox(np.array(ann['bbox']), img_width, img_height)
            if bbox is None: continue

            center, scale = box2cs(bbox,self.aspect_ratio,self.bbox_enlarge_factor)
            #root_cam = joint_cam[self.root_idx]

            data.append({
                'img_path': img_path,
                'img_id': image_id,
                'center': center,
                'scale': scale,
                'bbox': bbox,
                'joints': joint_img,  # [org_img_x, org_img_y, depth - root_depth]
                'joints_vis': joint_vis,
                'frame_id': frame_idx,
                'action_id': action_idx,
                'subaction_id': subaction_idx,
                'cam_id': cam_idx})
        return data

    def _load_detection_results(self):
        logger = logging.getLogger(__name__)
        logger.info("=> Load bbox file from {}".format(self.bbox_file))
        all_boxes = read_json_from_file(self.bbox_file)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_data = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = det_res['image_name']
            box = det_res['bbox']  # xywh
            score = det_res['score']
            nframes = det_res['nframes']
            frame_id = det_res['frame_id']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = box2cs(box, self.aspect_ratio, self.bbox_enlarge_factor)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_data.append({
                'image': osp.join(self.img_dir, img_name),
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'nframes': nframes,
                'frame_id': frame_id,
            })
        # logger.info('=> Total boxes: {}'.format(len(all_boxes)))
        # logger.info('=> Total boxes after filter low score@{}: {}'.format(self.image_thre, num_boxes))

        table_header = ["Total boxes", "Filter threshold", "Remaining boxes"]
        table_data = [[len(all_boxes), self.image_thre, num_boxes]]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Boxes Info Table: \n" + colored(table, "magenta"))

        return kpt_data

    def evaluate(self, cfg, preds, output_dir, boxes, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("=> Start evaluate")
        if self.phase == 'validate':
            output_dir = osp.join(output_dir, 'val_set_json_results')
        else:
            output_dir = osp.join(output_dir, 'test_set_json_results')

        create_folder(output_dir)

        ### processing our preds
        gts = self.data
        sample_num = len(gts)
        error = np.zeros((sample_num,self.num_joints))
        error_action = [[] for _ in range(len(self.action_name))]  # error for each sequence

        pred_save = []

        for n in range(sample_num):
            gt = gts[n]
            img_path = gt['img_path']
            img_id = gt['img_id']

            center = gt['center']
            scale = gt['scale']
            bbox = gt['bbox']
            joints = gt['joints']
            joints_vis = gt['joints_vis']
            frame_id = gt['frame_id']
            action_id = gt['action_id']
            subaction_id = gt['subaction_id']
            cam_id = gt['cam_id']

            # restore coordinates to original space
            pred_2d_kpt = preds[n, :, 0:2].copy()

            vis = False
            if vis:
                cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(n)
                tmpimg = cvimg.copy().astype(np.uint8)
                tmpkps = np.zeros((3, self.joint_num))
                tmpkps[0, :], tmpkps[1, :] = pred_2d_kpt[:, 0], pred_2d_kpt[:, 1]
                tmpkps[2, :] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                cv2.imwrite(filename + '_output.jpg', tmpimg)

            # error calculate
            error[n] = np.sqrt(np.sum((pred_2d_kpt[:,0:2] - joints[:,0:2]) ** 2,1))
            action_idx = int(img_path[img_path.find('act') + 4:img_path.find('act') + 6]) - 2
            error_action[action_idx].append(error[n].copy())
            # prediction save
            pred_save.append({'image_id': img_id, 'img_path':img_path,'bbox': bbox.tolist(),'gt_joints': joints.tolist(),
                              'pred_joints': pred_2d_kpt.tolist(), 'frame_id': frame_id, "action_id":action_id,
                              'subaction_id': subaction_id, 'cam_id':cam_id})  # joint_cam is root-relative coordinate

        tot_err = np.mean(error)
        metric = 'PA MPJPE' if self.protocol == 1 else 'MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' error (' + metric + ') >> tot: %.2f\n' % (tot_err)

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)

        print(eval_summary)

        # prediction save
        output_path = output_dir+'/human36m_S11.json'
        # output_path = osp.join(result_dir, '11')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)
