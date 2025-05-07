import os
import copy
import cv2
import mmcv
from PIL import Image
import matplotlib
import torch
import numpy as np
from functools import reduce
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
from torchvision import transforms
from torchvision.transforms import Resize
from mmcv.parallel.data_container import DataContainer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes, Box3DMode
# import sys;sys.path.append('/opt/data/private/aigc/MagicDrive_outdated')
# from magicdrive.runner.utils import show_box_on_views
# from magicdrive.dataset.utils import _preprocess_bbox

def box_center_shift(bboxes: LiDARInstance3DBoxes, new_center):
    raw_data = bboxes.tensor.numpy()
    new_bboxes = LiDARInstance3DBoxes(
        raw_data, box_dim=raw_data.shape[-1], origin=new_center)
    return new_bboxes

def create_heatmap_gt(bboxes_3d_input, lidar2image, metas, resolution=None):
    res=[]# each sample has different box num, could not stack across samples
    cam=lidar2image
    box,mask=bboxes_3d_input['bboxes'],bboxes_3d_input['masks']
    #box:bs,cam,n_box,8,3; mask:bs,cam,n_box; cam:bs,cam,4,4
    for batch in range(box.shape[0]):
        res.append(process_one_sample(box[batch],mask[batch],cam[batch].data, metas[batch], resolution))
    return torch.stack(res)

def process_one_sample(box,mask,cam, metas, resolution):
    res=[]
    for view in range(box.shape[0]):
        res.append(process_one_view_test(box[view],mask[view],cam[view],view, metas, resolution))
    return torch.stack(res)

def process_one_view(box,mask,cam,view, metas):
    res=[]
    for instance in range(box.shape[0]):
        res.append(process_one_instance(box[instance],mask[instance],cam,view,instance, metas).squeeze(0))
    # return torch.cat(res,dim=0)
    return reduce(torch.logical_or, res).to(torch.float32) # gather all the masks within the same view

def process_one_instance(corners,is_foreground,transform,view,instance, metas):    
    if not is_foreground:
        mask_width, mask_height = 50, 28
        mask = np.full((mask_height, mask_width), 0.0)
        return torch.tensor(mask.astype(np.float32)).unsqueeze(0)

    mask_width, mask_height = 50, 28
    mask = np.zeros((mask_height, mask_width))

    corners=corners.numpy()
    num_dots=corners.shape[0] # 8 corners

    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_dots, 1))], axis=-1
    )
    transform = copy.deepcopy(transform).reshape(4, 4).numpy()
    coords = coords @ transform.T
    coords = coords.reshape(-1, 4) # num_dots,4
    # if len(coords)==0:
    #     print('error, coords:',coords.shape)
    indices = coords[..., 2] > 0 # filter z > 0
    coords = coords[indices]

    # coords = coords.reshape(-1, 4)
    coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)#;print(coords.shape)
    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]
    coords[:, 0] /= 32  # 1600/50=32 
    coords[:, 1] /= 32.1428571429  # 900/28
    coords=coords.astype(np.int)

    coords = coords[..., :2].reshape(-1, 2)
    try:
        hull=ConvexHull(coords)
        coords = coords[hull.vertices]
    except:
        pass #input is less than 2-dimensional since all points have the same x coordinate, cannot yield hull
    # print(coords.shape)
    polygon = matplotlib.patches.Polygon(coords, closed=True)

    max_x,max_y=0,0
    if len(coords)==0:
        print('error, coords:',coords.shape,metas.data['token'],view,instance)
    for _x in range(mask_width):
        for _y in range(mask_height):
            if polygon.contains_point((_x,_y), radius=0):
                mask[_y, _x] = 1 #note that mask has its height dim in the front 

    tensor = torch.tensor(mask.astype(np.float32)).unsqueeze(0)
    return tensor


def process_one_view_test(box,mask,cam,view, metas, resolution):
    res=[]
    for instance in range(box.shape[0]):
        res.append(process_one_instance_test(box[instance],mask[instance],cam,view,instance, metas, resolution).squeeze(0))
    return torch.stack(res,dim=0).max(0).values
    # return reduce(torch.logical_or, res).to(torch.float32) # gather all the masks within the same view

def process_one_instance_test(corners,is_foreground,transform,view,instance, metas, resolution):
    mask_width, mask_height = resolution # resolution=w//8,h//8
    if not is_foreground:
        # mask_width, mask_height = 50, 28
        mask = np.full((mask_height, mask_width), 0.0)
        return torch.tensor(mask.astype(np.float32)).unsqueeze(0)

    # mask_width, mask_height = 50, 28
    mask = np.zeros((mask_height, mask_width))

    corners=corners.numpy()
    num_dots=corners.shape[0] # 8 corners

    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_dots, 1))], axis=-1
    )
    transform = copy.deepcopy(transform).reshape(4, 4).numpy()
    coords = coords @ transform.T
    coords = coords.reshape(-1, 4) # num_dots,4
    # if len(coords)==0:
    #     print('error, coords:',coords.shape)
    indices = coords[..., 2] > 0 # filter z > 0
    coords = coords[indices]

    # coords = coords.reshape(-1, 4)
    coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)#;print(coords.shape)
    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]
    # coords[:, 0] /= 32  # 1600/50=32 
    # coords[:, 1] /= 32.1428571429  # 900/28
    coords[:, 0] *= (mask_width/1600)  # 96/1600=0.06 
    coords[:, 1] *= (mask_height/900)  # 54/900=0.06
    coords=coords.astype(np.int)

    coords = coords[..., :2].reshape(-1, 2)
    try:
        hull=ConvexHull(coords)
        coords = coords[hull.vertices]
    except:
        pass #input is less than 2-dimensional since all points have the same x coordinate, cannot yield hull
    # print(coords.shape)
    polygon = matplotlib.patches.Polygon(coords, closed=True)

    max_x,max_y=0,0
    if len(coords)==0:
        print('error, coords:',coords.shape,metas.data['token'],view,instance)
    area_cnt=0
    for _x in range(mask_width):
        for _y in range(mask_height):
            if polygon.contains_point((_x,_y), radius=0):
                mask[_y, _x] = 1 #note that mask has its height dim in the front 
                area_cnt+=1
    # mask *= (area_cnt)**(-1)
    # mask *= 1-(area_cnt)/(28*50)
    mask *= 1-(area_cnt)/(mask_width*mask_height)
    tensor = torch.tensor(mask.astype(np.float32)).unsqueeze(0)
    return tensor
