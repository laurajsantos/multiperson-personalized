
"""
Demo code

Example usage:

python3 tools/demo.py configs/smpl/tune.py ./demo/raw_teaser.png --ckpt /path/to/model
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn

import argparse
import os.path as osp
import sys
import cv2
import numpy as np
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_PATH)
from mmcv import Config
from mmcv.runner import Runner

from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel
from mmdet.apis.train import build_optimizer
from mmdet.models.utils.smpl.renderer import Renderer
from mmdet import __version__
from mmdet.models import build_detector
from mmdet.datasets.transforms import ImageTransform
from mmdet.datasets.utils import to_tensor


denormalize = lambda x: x.transpose([1, 2, 0]) * np.array([0.229, 0.224, 0.225])[None, None, :] + \
                        np.array([0.485, 0.456, 0.406])[None, None,]

# dataset settings
bone_list = [[0, 1], [1, 2], [2, 14], [14, 3], 
                        [3, 4], [4, 5], [14, 16], [16, 15], 
                        [15, 12], [12, 8], [8, 7], [7, 6],
                        [12, 9], [9, 10], [10, 11], [12, 17], [17, 18], [18, 19], 
                        [13, 19], [19, 21], [21, 23], [19, 20], [20, 22]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
adsta=5320
adsto=5330
chsta = 0
chsto = 0
dist =2.5
lad = []
lch = []
def plot_skeletons(fimg, joints_2d, flag):
    x = joints_2d[:, 0]
    y = joints_2d[:, 1]
    #x0 = joints_2d[0,:,0]
    #y0 = joints_2d[0,:,1]
    #x1 = joints_2d[1,:,0]
    #y1 = joints_2d[1,:,1]
    if flag == 'adult':
        color = (0,0,255)
        color1 = (255, 0, 0)
    else:
        color = (255,0,0)
    for ib,bone in enumerate(bone_list):
        x_0 = int((x[bone[0]]))
        x_1 = int((x[bone[1]]))
        y_0 = int((y[bone[0]]))
        y_1 = int((y[bone[1]]))
        #print('point',[x_0,y_0,x_1,y_1])
        cv2.line(fimg, (x_0,y_0), (x_1,y_1), color, 2)
        cv2.drawMarker(fimg, (x_0, y_0), color)
        cv2.drawMarker(fimg, (x_1, y_1), color)
    return fimg

def perspective_projection_altern(points, rotation, translation, focal_length, camera_center):
    K = np.zeros([3,3])
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[2,2] = 1
    #print(camera_center)
    K[:-1, -1] = camera_center
    #print(K)
    #print(rotation.shape)
    #print(points.shape)
    pp = []
    for ip,epoints in enumerate(points):
        points_rot = np.einsum('ij,kj->ki', rotation,epoints)
        #print(translation.shape)
        #print(translation[ip])
        points_t = points_rot + translation[ip]

        projected_points = points_t / points_t[:,-1, None]

        projected_points = np.einsum('ij,kj->ki', K, projected_points)
        pp.append(projected_points[:,0:2])
    pp = np.asarray(pp)
    #print(pp.shape)
    return pp

def renderer_bv(img_t, verts_t, trans_t, bboxes_t, focal_length, render):
    R_bv = torch.zeros(3, 3)
    R_bv[0, 0] = R_bv[2, 1] = 1
    R_bv[1, 2] = -1
    bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
    area_mask = torch.tensor(bbox_area > bbox_area.max() * 0.05)
    verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]
    verts_t = verts_t + trans_t.unsqueeze(1)
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
    p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    p_center = 0.5 * (p_min + p_max)
    # trans_tr = torch.einsum('bj,kj->bk', trans_t, R_bv)
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
            verts_tfar.view(-1, 3) - p_center).max(0)[0]
    h, w = img_t.shape[-2:]
    # h, w = min(h, w), min(h, w)
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
        dis_min[2])
    z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
        dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
    img_right = render([torch.ones_like(img_t)], [verts_right],
                       translation=[torch.zeros_like(trans_t)])
    return img_right[0]

def getredObj(rgbframe):
    lowerBound1 = np.array([90, 140, 0]) ## Blue Color
    upperBound1 = np.array([150, 255, 174])

    #lowerBound1 = np.array([0, 50, 0])
    #upperBound1 = np.array([10, 255, 170])
    #print('RGB: ', rgbframe)
    #bgrframe = cv2.cvtColor(rgbframe, cv2.COLOR_RGB2BGR)
    hsvframe=cv2.cvtColor(rgbframe, cv2.COLOR_BGR2HSV)

    mask1=cv2.inRange(hsvframe,lowerBound1,upperBound1) #color filtering
    mask = mask1
    aux = np.where(mask != 0)[0]
    #print('Mask ',aux)
    kernelOpen=np.ones((15,15))
    kernelClose=np.ones((40,40))
    # to delete noise from images
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen) 
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    #print('Mask Fim: ', np.where(maskClose != 0)[0])
    # to find contours of the red image
    # cv2.RETR_EXTERNAL = to get external contours only
    # cv2.CHAIN_APPROX_NONE = all the boundary points are stored
    # conts is an array containing the coordinates of all contours of the obj
    conts,hl=cv2.findContours(maskClose.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

    # Cycle through contours and add area to array
    areas = []
    for c in conts:
        areas.append(cv2.contourArea(c)) #returns area of each red contours and number of non-zero pixels

    # Sort array of areas by size
    #sorted (iterable, key=key, reverse=reverse)
    #iterable= the sequence to sort, list, tuple ecc
    #key= a function to execute to decide the order, default is none
    # reverse=  boolean, false will sort ascending, true will sort descending
    sorted_areas = sorted(zip(areas, conts), key=lambda x: x[0], reverse=True) #takes areas as reference x(0) not conts
    #print('Red obj: ',sorted_areas)

    if len(sorted_areas) >= 1:
        # Find nth largest using data[n-1][1]
        return sorted_areas[0]
    else:
        return None

    count = count+1

    return conts

def getorangeObj(rgbframe):
    lowerBound1 = np.array([0, 170, 170]) ## Orange Color
    upperBound1 = np.array([17, 255, 255])

    #lowerBound1 = np.array([0, 50, 0])
    #upperBound1 = np.array([10, 255, 170])
    #print('RGB: ', rgbframe)
    #bgrframe = cv2.cvtColor(rgbframe, cv2.COLOR_RGB2BGR)
    hsvframe=cv2.cvtColor(rgbframe, cv2.COLOR_BGR2HSV)

    mask1=cv2.inRange(hsvframe,lowerBound1,upperBound1) #color filtering
    mask = mask1
    aux = np.where(mask != 0)[0]
    #print('Mask ',aux)
    kernelOpen=np.ones((3,3))
    kernelClose=np.ones((20,20))
    # to delete noise from images
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen) 
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    #print('Mask Fim: ', np.where(maskClose != 0)[0])
    # to find contours of the red image
    # cv2.RETR_EXTERNAL = to get external contours only
    # cv2.CHAIN_APPROX_NONE = all the boundary points are stored
    # conts is an array containing the coordinates of all contours of the obj
    conts,hl=cv2.findContours(maskClose.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

    # Cycle through contours and add area to array
    areas = []
    for c in conts:
        areas.append(cv2.contourArea(c)) #returns area of each red contours and number of non-zero pixels

    # Sort array of areas by size
    #sorted (iterable, key=key, reverse=reverse)
    #iterable= the sequence to sort, list, tuple ecc
    #key= a function to execute to decide the order, default is none
    # reverse=  boolean, false will sort ascending, true will sort descending
    sorted_areas = sorted(zip(areas, conts), key=lambda x: x[0], reverse=True) #takes areas as reference x(0) not conts
    #print('Red obj: ',sorted_areas)

    if len(sorted_areas) >= 1:
        # Find nth largest using data[n-1][1]
        return sorted_areas[0]
    else:
        return None
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length
def getActive(poses, rect):

    ither = None
    ichil = None

    def size(vector):
        return np.sqrt(sum(x**2 for x in vector))

    def distance(vector1, vector2):
        return size(vector1 - vector2)

    def distances(array1, array2):
        return [distance(array1[i], array2[i]) for i in range(0,len(array1))]

    def areacalc(a, b):  # returns None if rectangles don't intersect
    #intersected area btw rectangle and trunk, see below
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if (dx>=0) and (dy>=0):
            return dx*dy # intersected area
        else:
            return 0

    

    # Extraction of the area of the rectangle and selection of the red element with the maximal area
    x,y,w,h=cv2.boundingRect(rect[1]) # rect[1] contains contours

    redshirtcomp=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
    sumwh=w+h # normalization factor
    redshirt=np.array([x,y,x+w,y+h])

    # Initialization of the variables for the centroid distances and overlap areas
    dlist=[]
    alist=[]
    ilist=[]
    shoulders = []

    for pose in range(len(poses)): # i: index for numbers, pose: index for skeletons
    # poses in the generale definition of the function is equal to raw_poses when the getActive is called
        # Get joints for the first body
        #joints = pose.joints
        joint_points = poses[pose]

        shoulders.append(joint_points[8,0])
        trunk = np.array([joint_points[8,0],
                        joint_points[8,1],
                        joint_points[3,0],
                        joint_points[3,1]])
        trunkcomp = np.array([[joint_points[8,0], joint_points[8,1]],
                            [joint_points[9,0], joint_points[9,1]],
                            [joint_points[1,0], joint_points[1,1]],
                            [joint_points[4,0], joint_points[4,1]]])
        centroidtrunk = centeroidnp(trunkcomp)
        centroidred = centeroidnp(redshirtcomp)

        #fdists=distances(np.array(centroidtrunk), np.array(centroidred))
        fdists = np.sqrt(sum((np.array(centroidtrunk) - np.array(centroidred))**2))
        #print(fdists)
        farea =areacalc(trunk,redshirt) # intersected area btw rectangle and trunk (trunk is defined by coordinates)
        #fdists=distance(centroidtrunk, centroidred)
        #fdists=max(distances(np.array([centroidtrunk,centroidtrunk,centroidtrunk,centroidtrunk]),redshirtcomp)) # calculates the distances btw the center point and the coordinates of the rectangle

        alist.append(farea)
        ilist.append(pose)
        dlist.append(fdists)

    if len(dlist)==1:
        ither = 0
        #return poses[0], None
        return 0

    sortedshoulders = np.argsort(np.asarray(shoulders))
    sortedpos=np.argsort(alist) # argsort returns an array of indices of the sorted array

    dlistchosen = np.array(dlist)[sortedshoulders[-2:].astype(int)]
    ilistchosen = np.array(ilist)[sortedshoulders[-2:].astype(int)]
    sorteddist=np.argsort(dlistchosen)
    ither = ilistchosen[sorteddist[0]]
    ichil = ilistchosen[sorteddist[1]]

    positions = [ither, ichil]
    #return poses[sortedshoulders[-2:][sorteddist[0]]], poses[sortedshoulders[-2:][sorteddist[1]]]
    return positions

def getActiveNAO(poses, rect):

    # Extraction of the area of the rectangle and selection of the red element with the maximal area
    x,y,w,h=cv2.boundingRect(rect[1]) # rect[1] contains contours

    #print("rect_orange", x, y, w, h)
    redshirt=np.array([x,y,x+w,y+h])
    redshirtcomp=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])

    dlist=[]
    ilist=[]

    for pose in range(len(poses)): # i: index for numbers, pose: index for skeletons
    # poses in the generale definition of the function is equal to raw_poses when the getActive is called
        # Get joints for the first body
        #joints = pose.joints
        joint_points = poses[pose]

        head = joint_points[18,0], joint_points[18,1]
        headtop = joint_points[13,0], joint_points[13,1]
 
        if (redshirt[2] >= head[0] and head[0] >= redshirt[0] and redshirt[3] >= head[1] and head[1] >= redshirt[1]) or (redshirt[2] >= headtop[0] and headtop[0] >= redshirt[0] and redshirt[3] >= headtop[1] and headtop[1] >= redshirt[1]):
            #print(redshirt, head)
            positions = pose
            #print("Delete NAO")
            return positions
        #else:
            
            #trunkcomp = np.array([[joint_points[8,0], joint_points[8,1]],
                                #[joint_points[9,0], joint_points[9,1]],
                                #[joint_points[1,0], joint_points[1,1]],
                                #[joint_points[4,0], joint_points[4,1]]])
            #centroidtrunk = centeroidnp(trunkcomp)
            #centroidred = centeroidnp(redshirtcomp)

            #fdists = np.sqrt(sum((np.array(centroidtrunk) - np.array(centroidred))**2))
        
            #ilist.append(pose)
            #dlist.append(fdists)
    #dlistchosen = np.array(dlist)
    #ilistchosen = np.array(ilist)
    #sorteddist=np.argsort(dlistchosen)
    #irob = ilistchosen[sorteddist[0]]
    #return irob

def save_joints_2px(joints, flag, dimension, filename, output_folder, FOCAL_LENGTH):
    if len(joints[0]) != dimension:
        joints = joints[0]
    if flag == 'child':
        if dimension == 3:
            f = open(output_folder + '/results_child'+ str(FOCAL_LENGTH) +'.txt', 'a')
        elif dimension == 2:
            f = open(output_folder + '/results_child2d' + str(FOCAL_LENGTH) + '.txt', 'a')
        line = filename + " " + " ".join(str(r) for v in joints for r in v) + "\n"
        f.write(line)
        f.close()
    elif flag == 'adult':
        if dimension == 3:
            f = open(output_folder + '/results_adult'+ str(FOCAL_LENGTH) +'.txt', 'a')
        elif dimension == 2:
            f = open(output_folder + '/results_adult2d' + str(FOCAL_LENGTH) + '.txt', 'a')
        line = filename + " " + " ".join(str(r) for v in joints for r in v) + "\n"
        f.write(line)
        f.close()

def save_joints(joints, joints_2d, bboxes, filename, output_folder):
    """
    f = open(output_folder+'/results.txt','a')
    print(output_folder+'/results.txt')
    if joints.shape[0]!=24:
        for ij in range(0, joints.shape[0]):
            line = filename+" "+str(ij)+" "+" ".join(str(r) for v in joints[ij] for r in v) + "\n"
            f.write(line)
    f.close()
    """
    f = open(output_folder+'/results_2d.txt','a')
    #print(output_folder+'/results_2d.txt')
    if joints_2d.shape[0]!=24:
        for ij in range(0, joints_2d.shape[0]):
            line = filename+" "+str(ij)+" "+" ".join(str(r) for v in joints_2d[ij] for r in v) + "\n"
            f.write(line)
    f.close()
    f = open(output_folder+'/bboxes.txt','a')
    #print(output_folder+'/bboxes.txt')
    for ij in range(0, bboxes.shape[0]):
        line = filename+" "+str(ij)+" "+" ".join(str(v) for v in bboxes[ij]) + "\n"
        f.write(line)
    f.close()

def prepare_dump(pred_results, img, render, bbox_results, FOCAL_LENGTH,pose, jnao, nframe, output_folder, flag):
    verts = pred_results['pred_vertices'] + pred_results['pred_translation'][:, None]
    # 'pred_rotmat', 'pred_betas', 'pred_camera', 'pred_vertices', 'pred_joints', 'pred_translation', 'bboxes'
    bboxes_size = []
    for i in range(0, pred_results['bboxes'].shape[0]):
        msize = max([abs(pred_results['bboxes'][i, 0] - pred_results['bboxes'][i, 2]),abs(pred_results['bboxes'][i, 1] - pred_results['bboxes'][i, 3])])
        bboxes_size.append(msize)
    focal = pred_results['pred_translation'][0][2]*(1e-6 + pred_results['pred_camera'][0][0]*bboxes_size[0])
    camera = pred_results['pred_camera']
    print('camerabef', camera)
    if jnao is not None:
        camwoNAO = torch.cat((camera[:jnao],camera[jnao+1:]))
        bboxes_size = np.delete(bboxes_size,jnao,0)
    else:
        camwoNAO = camera
    print('cameraaft', camwoNAO)
    print('bbox_bef', bboxes_size)
    print('bbox_aft', bboxes_size)
    #focal_new = 1.9*(1e-6 + pred_results['pred_camera'][0][0]*bboxes_size[0])
    focal_new = dist/2*(1e-6 + camwoNAO[pose[0]][0]*bboxes_size[pose[0]])
    focal_new_2 = dist/2*(1e-6 + camwoNAO[pose[1]][0]*bboxes_size[pose[1]]) 
    f = open(output_folder+'/focal_length.txt','a')
    #print(output_folder+'/bboxes.txt')
    line = flag + " " + str(nframe)+" "+str(dist)+" "+str(focal_new.cpu().detach().numpy())+" "+str(focal_new_2.cpu().detach().numpy())+" "+str(bboxes_size[pose[0]])+" "+str(bboxes_size[pose[1]])+" "+str(camwoNAO[pose[0]][0].cpu().detach().numpy())+" "+str(camwoNAO[pose[1]][0].cpu().detach().numpy())+" " + "\n"
    f.write(line)
    f.close()
    if flag =="adult":
        if bboxes_size[pose[0]] > bboxes_size[pose[1]]:
            lad.append(focal_new.cpu().detach().numpy())
        elif bboxes_size[pose[0]] < bboxes_size[pose[1]]:
            lad.append(focal_new_2.cpu().detach().numpy())
    else:
        if bboxes_size[pose[0]] > bboxes_size[pose[1]]:
            lch.append(focal_new_2.cpu().detach().numpy())
        elif bboxes_size[pose[0]] < bboxes_size[pose[1]]:
            lch.append(focal_new.cpu().detach().numpy())
    print("New focal length - ", focal_new, focal_new_2, camwoNAO[pose[0]], camwoNAO[pose[1]])
    print("BBox 1 - ", bboxes_size[pose[0]], focal_new)
    print("BBox 2 - ", bboxes_size[pose[1]], focal_new_2)
    pred_trans = pred_results['pred_translation'].cpu()
    pred_camera = pred_results['pred_camera'].cpu()
    pred_betas = pred_results['pred_betas'].cpu()
    pred_rotmat = pred_results['pred_rotmat'].cpu()
    pred_verts = pred_results['pred_vertices'].cpu()
    bboxes = pred_results['bboxes']
    #print('focal',focal)
    #print('new_focal',focal_new)
    #print('bbox_size',bboxes_size)
    #print('bbox',pred_results['bboxes'])
    #print('param_cam',pred_results['pred_camera'])
    #print('pred_translation',pred_results['pred_translation'])
    img_bbox = img.copy()
    for bbox in bboxes:
        img_bbox = cv2.rectangle(img_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    img_th = torch.tensor(img_bbox.transpose([2, 0, 1]))
    _, H, W = img_th.shape
    try:
        fv_rendered = render([img_th.clone()], [pred_verts], translation=[pred_trans])[0]
        bv_rendered = renderer_bv(img_th, pred_verts, pred_trans, bbox_results[0], FOCAL_LENGTH, render)
    except Exception as e:
        print(e)
        return None

    total_img = np.zeros((3 * H, W, 3))
    total_img[:H] += img
    total_img[H:2 * H] += fv_rendered.transpose([1, 2, 0])
    total_img[2 * H:] += bv_rendered.transpose([1, 2, 0])
    total_img = (total_img * 255).astype(np.uint8)
    return total_img

def new_depth(pred_results, FOCAL_LENGTH, param, img):
    translation = pred_results['pred_translation'].clone().detach()
    for i in range(0, pred_results['bboxes'].shape[0]):
        msize = max([abs(pred_results['bboxes'][i, 0] - pred_results['bboxes'][i, 2]),abs(pred_results['bboxes'][i, 1] - pred_results['bboxes'][i, 3])])
        depth = param * FOCAL_LENGTH / (1e-6 + pred_results['pred_camera'][i][0]*msize)
        translation[i, -1] = depth
    return translation


def calculate_joints2d(pred_results, img, FOCAL_LENGTH):
    pred_joints = pred_results['pred_joints'].detach().cpu().numpy()
    pred_trans = pred_results['pred_translation'].detach().cpu().numpy()

    rotation_Is = np.eye(3)
    focal_length = FOCAL_LENGTH
    bboxes = pred_results['bboxes']
    img_size = np.asarray([img.shape[1], img.shape[0]])
    #print('SHAPE 2: ', img.shape)

    pred_keypoints_2d_smpl = perspective_projection_altern(pred_joints, rotation_Is, pred_trans, focal_length, img_size / 2)
    batch_size = pred_joints.shape[0]

    return pred_keypoints_2d_smpl

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--image_folder', help='Path to folder with images')
    parser.add_argument('--output_folder', default='model_output', help='Path to save results')
    parser.add_argument('--ckpt', type=str, default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print("Entrei")
    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.ckpt:
        cfg.resume_from = args.ckpt

    cfg.test_cfg.rcnn.score_thr = 0.5
    #import pdb; pdb.set_trace()
    FOCAL_LENGTH = cfg.get('FOCAL_LENGTH', 389)
    FOCAL_LENGTH = 389
    cfg.FOCAL_LENGTH = 389
    #print('fl_1',cfg.FOCAL_LENGTH)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=('Human',))
    # add an attribute for visualization convenience
    model.CLASSES = ('Human',)
    #print('fl_2', cfg.FOCAL_LENGTH)
    model = MMDataParallel(model, device_ids=[0]).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = Runner(model, lambda x: x, optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.resume(cfg.resume_from)
    model = runner.model
    model.eval()

    render = Renderer(focal_length=FOCAL_LENGTH)
    img_transform = ImageTransform(
            size_divisor=32, **img_norm_cfg)
    img_scale = cfg.common_val_cfg.img_scale
    with torch.no_grad():
        folder_name = args.image_folder
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        images = os.listdir(folder_name)
        outfile = output_folder+ '/results_video.avi'
        
        images = os.listdir(folder_name)
        vid = [s for s in images if 'video1' in s]
        print(vid)
        inp = cv2.VideoCapture(folder_name+'/'+vid[0])
        nframe = 0
        ret, image = inp.read()
        w = int(inp.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(inp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(outfile, fourcc, 30, (832,512))
        count_blue_none = 0
        count_orange_none = 0
        count_3skeletons = 0
        while ret:
            #print('nframe', nframe)
            cfg.FOCAL_LENGTH = 430
            FOCAL_LENGTH = 430
            #389
        
            file_name = str(nframe)
            img = image
            #print('INICIO', file_name)
            ori_shape = img.shape

            img, img_shape, pad_shape, scale_factor = img_transform(img, img_scale)

            # Force padding for the issue of multi-GPU training
            padded_img = np.zeros((img.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
            padded_img[:, :img.shape[-2], :img.shape[-1]] = img
            img = padded_img

            assert img.shape[1] == 512 and img.shape[2] == 832, "Image shape incorrect"
            #print('SHAPE 1: ', img.shape)
            #import pdb; pdb.set_trace()
            ret, image = inp.read()
            nframe = nframe + 1
            param = 2
            fixed_DC =DC([to_tensor(img[None, ...])], stack=True)
            img = denormalize(img)
            final_img = (img * 255).astype(np.uint8)
            fimg = final_img[:,:, ::-1].copy()
            joints_NAO = None
            rect = getredObj(fimg)
            rect_orange = getorangeObj(fimg)
            
            #if rect is not None:
                #print('rect blue')
                
            data_batch = dict(
                img=fixed_DC,
                img_meta=DC([{'img_shape':img_shape, 'scale_factor':scale_factor, 'flip':False, 'ori_shape':ori_shape, 'focal_length':FOCAL_LENGTH, 'param':param}], cpu_only=True),
                )
            #print('FL: ', cfg.FOCAL_LENGTH)
            bbox_results, pred_results = model(**data_batch, return_loss=False)

            if pred_results is not None:
                
                #print('Juntas: ', joints_3d.shape[0])
                pred_results['bboxes'] = bbox_results[0]
                
                #print(file_name)
                bbox_real = pred_results['bboxes']
                joints_2d = calculate_joints2d(pred_results, img, FOCAL_LENGTH)
                #fimg = plot_skeletons(fimg, joints_2d[0,:,:], 'child')
                #if np.size(joints_2d, 0) == 2:
                #    fimg = plot_skeletons(fimg, joints_2d[1,:,:], 'adult')
                #save_joints(joints_3d, joints_2d, bbox_real, file_name, output_folder)
                mode_detection = 'blue' #'blue' or 'pose'
                if mode_detection == 'blue':
                    # Se existir azul
                    if rect is None:
                        count_blue_none = count_blue_none + 1
                    if rect_orange is None:
                        count_orange_none = count_orange_none + 1
                    if rect is not None:
                        # Se existir pelo menos um esqueleto e o laranja do NAO, confirmar se algum esqueleto é do NAO
                        if rect_orange is not None:
                            joints_NAO = getActiveNAO(joints_2d, rect_orange)
                            #print("joints", joints_2d)
                            if joints_NAO is not None:
                                joints_2d = np.delete(joints_2d,joints_NAO,0)
                        if len(joints_2d) > 1:
                            #print('duasjuntas')
                            if len(joints_2d) > 2:
                                count_3skeletons = count_3skeletons + 1
                            pose = getActive(joints_2d, rect)
                            x,y,w,h=cv2.boundingRect(rect[1])
                            #print('Retangulo: ', x, y, w, h)
                            # Only 1 skeleton
                            #print('2people')
                            joints_translated = pred_results['pred_joints']+ pred_results['pred_translation'][:,None]
                            #print('Translation: ', pred_results['pred_translation'])
                            joints_3d = joints_translated.cpu().detach().numpy()
                            if joints_NAO is not None:
                                joints_3d = np.delete(joints_3d,joints_NAO,0)
                            fimg = plot_skeletons(fimg, joints_2d[pose[0]], 'adult')
                            fimg = plot_skeletons(fimg, joints_2d[pose[1]], 'child')
                            # Calculate new focal length
                            if (nframe >= adsta and adsto >= nframe):
                                print('nframe',nframe)
                                print(joints_NAO)
                                img_viz = prepare_dump(pred_results, img, render, bbox_results, FOCAL_LENGTH, pose,joints_NAO, nframe, output_folder,'adult')
                                cv2.imwrite(output_folder + '/' + str(nframe)+ str(FOCAL_LENGTH)+'.jpg', img_viz[:, :, ::-1])
                            if (nframe >= chsta and chsto >= nframe):
                                print('nframe',nframe)
                                img_viz = prepare_dump(pred_results, img, render, bbox_results, FOCAL_LENGTH, pose,joints_NAO, nframe, output_folder,'child')

                            if (nframe > chsto and nframe > adsto):
                                break
                            
            out.write(fimg)
        ladsp = np.asarray(lad)
        lchsp = np.asarray(lch)
        mediana = np.median(ladsp)
        medianc = np.median(lchsp)
        if ladsp.size >0:
            pc25a = np.percentile(ladsp,25)
            pc75a = np.percentile(ladsp,75)
        else:
            pc25a = 0
            pc75a = 0
        if lchsp.size>0:
            pc25c = np.percentile(lchsp,25)
            pc75c = np.percentile(lchsp,75)
        else:
            pc25c = 0
            pc75c = 0
        print(output_folder)
        print('Child focal length:', medianc, pc25c, pc75c)
        print('Adult focal length:',mediana, pc25a, pc75a)
        f = open(output_folder+'/mediana.txt','a')
        line = output_folder+' '+str(adsta)+' '+ str(adsto)+' '+str(chsta)+' '+str(chsto)+' '+str(mediana)+' '+str(medianc)+' '+str(pc25a)+' '+str(pc25c)+' '+str(pc75a)+' '+str(pc75c)
        f.write(line)
        f.close()
        print("Orange none", count_orange_none)
        print("Blue none", count_blue_none)
        print("3 skeletons", count_3skeletons)
        out.release()
        inp.release()


if __name__ == '__main__':
    main()
