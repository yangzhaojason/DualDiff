from typing import Tuple, List
from functools import partial
import random

import torch
import numpy as np
from torchvision import transforms
from einops import rearrange

from transformers import CLIPTokenizer
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from ..runner.utils import trans_boxes_to_views
from ..networks.utils import *

META_KEY_LIST = [
    "gt_bboxes_3d",
    "gt_labels_3d",
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "metas",
]


def _tokenize_captions(examples, template, tokenizer=None, is_train=True, txt_aug=None):
    captions = []
    for example in examples:
        caption = template.format(**example["metas"].data)
        if isinstance(txt_aug, list): # use_aug_text
            for i in range(6):
                tmp = caption+' '+txt_aug.pop(0).capitalize()+'.' if txt_aug else caption
                captions.append(tmp)
        else:
            captions.append(caption)
    assert not txt_aug # None or empty list
    captions.append("")
    if tokenizer is None:
        return None, captions

    # pad in the collate_fn function
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="do_not_pad",
        truncation=True,
    )
    input_ids = inputs.input_ids
    # pad to the longest of current batch (might differ between cards)
    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids
    return padded_tokens, captions


def ensure_canvas(coords, canvas_size: Tuple[int, int]):
    """Box with any point in range of canvas should be kept.

    Args:
        coords (_type_): _description_
        canvas_size (Tuple[int, int]): _description_

    Returns:
        np.array: mask on first axis.
    """
    (h, w) = canvas_size
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    w_mask = np.any(np.logical_and(
        coords[..., 0] > 0, coords[..., 0] < w), axis=1)
    h_mask = np.any(np.logical_and(
        coords[..., 1] > 0, coords[..., 1] < h), axis=1)
    c_mask = np.logical_and(c_mask, np.logical_and(w_mask, h_mask))
    return c_mask


def ensure_positive_z(coords):
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    return c_mask


def random_0_to_1(mask: np.array, num):
    assert mask.ndim == 1
    inds = np.where(mask == 0)[0].tolist()
    random.shuffle(inds)
    mask = np.copy(mask)
    mask[inds[:num]] = 1
    return mask


def _transform_all(examples, matrix_key, proj):
    """project all bbox to views, return 2d coordinates.

    Args:
        examples (List): collate_fn input.

    Returns:
        2-d list: List[List[np.array]] for B, N_cam. Can be [].
    """
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    # lidar2image (np.array): lidar to image view transformation
    trans_matrix = np.stack([example[matrix_key].data.numpy()
                            for example in examples], axis=0)
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = np.stack([example['img_aug_matrix'].data.numpy()
                               for example in examples], axis=0)
    B, N_cam = trans_matrix.shape[:2]

    bboxes_coord = []
    # for each keyframe set
    for idx in range(B):
        # if zero, add empty list
        if len(gt_bboxes_3d[idx]) == 0:
            # keep N_cam dim for convenient
            bboxes_coord.append([None for _ in range(N_cam)])
            continue

        coords_list = trans_boxes_to_views(
            gt_bboxes_3d[idx], trans_matrix[idx], img_aug_matrix[idx], proj)
        bboxes_coord.append(coords_list)
    return bboxes_coord


def _preprocess_bbox(bbox_mode, canvas_size, examples, is_train=True,
                     view_shared=False, use_3d_filter=True, bbox_add_ratio=0,
                     bbox_add_num=0, bbox_drop_ratio=0, for_mask=False):
    """Pre-processing for bbox
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
        view_shared: if enabled, all views share same set of bbox and output
            N_cam=1; otherwise, use projection to keep only visible bboxes.
    Return:
        in form of dict:
            bboxes (Tensor): B, N_cam, max_len, ...
            classes (LongTensor): B, N_cam, max_len
            masks: 1 for data, 0 for padding
    """
    # init data
    bboxes = []
    classes = []
    max_len = 0
    if for_mask:
        gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
            box_center_shift(example['gt_bboxes_3d'].data, (0.5, 0.5, 0.5)) for example in examples]
    else:
        gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
            example["gt_bboxes_3d"].data for example in examples]
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]

    # params
    B = len(gt_bboxes_3d)
    N_cam = len(examples[0]['lidar2image'].data.numpy())
    N_out = 1 if view_shared else N_cam

    bboxes_coord = None
    if not view_shared and not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True)
    elif not view_shared:
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False)

    # for each keyframe set
    for idx in range(B):
        bboxes_kf = gt_bboxes_3d[idx]
        classes_kf = gt_labels_3d[idx]

        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0 or (
                random.random() < bbox_drop_ratio and is_train):
            bboxes.append([None] * N_out)
            classes.append([None] * N_out)
            continue

        # whether share the boxes across views, filtered by 2d projection.
        if not view_shared:
            index_list = []  # each view has a mask
            if use_3d_filter:
                coords_list = bboxes_coord_3d[idx]
                filter_func = ensure_positive_z
            else:
                # filter bbox according to 2d projection on image canvas
                coords_list = bboxes_coord[idx]
                # judge coord by cancas_size
                filter_func = partial(ensure_canvas, canvas_size=canvas_size)
            # we do not need to handle None since we already filter for len=0
            for coords in coords_list:
                c_mask = filter_func(coords)
                if random.random() < bbox_add_ratio and is_train:
                    c_mask = random_0_to_1(c_mask, bbox_add_num)
                index_list.append(c_mask)
                max_len = max(max_len, c_mask.sum())
        else:
            # we use as mask, torch.bool is important
            index_list = [torch.ones(len(bboxes_kf), dtype=torch.bool)]
            max_len = max(max_len, len(bboxes_kf))

        # construct data
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        bboxes.append([bboxes_pt[ind] for ind in index_list])
        classes.append([classes_kf[ind] for ind in index_list])
        bbox_shape = bboxes_pt.shape[1:]

    # there is no (visible) boxes in this batch
    if max_len == 0:
        return None, None

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    for _b in range(B):
        _bboxes = bboxes[_b]
        _classes = classes[_b]
        for _n in range(N_out):
            if _bboxes[_n] is None:
                continue  # empty for this batch
            this_box_num = len(_bboxes[_n])
            ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
            ret_classes[_b, _n, :this_box_num] = _classes[_n]
            ret_masks[_b, _n, :this_box_num] = True

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict, bboxes_coord


def _preprocess_map_vec(cfg,bboxes,classes,view_shared):
    """
    process vectorized discretized map points just like _preprocess_box()
    """
    assert view_shared, 'we could only achieve view-shared'
    N_out = 1 # as view_shared=True in _preprocess_box() does
    B = len(classes) # batchsize
    # if there is an empty sample, it appears as a None in both box & cls, otherwise tensor
    max_len = max([len(i) for i in classes if i is not None]+[0])
    if max_len == 0: # there is no (visible) map in this batch
        return None # we don't have as well as don't use bboxes_coord
    #有些地图的vec为空，对应在传入的bboxes和classes中为None
    # if cfg.use_map_vec_40pts:
    #     bbox_shape = (40,3)
    # else:
    bbox_shape = (8,3)
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    for _b in range(B):
        _bboxes = bboxes[_b]
        _classes = classes[_b]
        for _n in range(N_out):
            if _bboxes is None:
                continue  # empty for this batch
            this_box_num = len(_bboxes)
            ret_bboxes[_b, _n, :this_box_num] = _bboxes
            ret_classes[_b, _n, :this_box_num] = _classes
            ret_masks[_b, _n, :this_box_num] = True

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict # we don't have as well as don't use bboxes_coord


def collate_fn(
    examples: Tuple[dict, ...],
    template: str,
    tokenizer: CLIPTokenizer = None,
    is_train: bool = True,
    bbox_mode: str = None,
    bbox_view_shared: bool = False,
    bbox_drop_ratio: float = 0,
    bbox_add_ratio: float = 0,
    bbox_add_num: int = 3,
    occ_proj = None,
    map_vec = None,
    cfg = None,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make masks (gt_masks_bev, gt_aux_bev) into tensor
        -> [N, 25 = 8 map + 10 obj + 7 aux, 200, 200]
    3. make caption (location, desctiption, timeofday) and tokenize, padding
        -> [N, pad_length]
    4. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    We keep other meta data as original.
    """
    if bbox_add_ratio > 0 and is_train:
        assert bbox_view_shared == False, "You cannot add any box on view shared."

    # mask
    # if "gt_aux_bev" in examples[0] and examples[0]["gt_aux_bev"] is not None:
    #     keys = ["gt_masks_bev", "gt_aux_bev"]
    #     assert bbox_drop_ratio == 0, "map is not affected in bbox_drop"
    # else:
    #     keys = ["gt_masks_bev"]
    # official weights expect 8 channel in gt_masks_bev, originally 18 channels(8 map + 10 obj) 
    keys = ["gt_masks_bev"]
    # fmt: off
    bev_map_with_aux = torch.stack([torch.from_numpy(np.concatenate([
        example[key][:8] for key in keys  # np array, channel-last
    ], axis=0)).float() for example in examples], dim=0)  # float32
    # fmt: on

    def hd_crop(x,in_shape=(432,768),out_shape=(256,704)): # (h,w)
        h_crop = in_shape[0]-out_shape[0]
        w_crop = (in_shape[1]-out_shape[1])//2
        y = x[...,(h_crop):,w_crop:(in_shape[1]-w_crop)]
        assert y.shape[-2:]==out_shape
        return y
    
    def pad_top(x,in_shape=(224,400),out_shape=(225,400)):
        out = torch.zeros((*x.shape[:-2],*out_shape))
        out[...,(out_shape[0]-in_shape[0]):,:] = x
        return out

    def crop_hd(x,in_shape):
        per_w=in_shape[2]//6
        res = [
            x[...,:per_w],
            x[...,per_w:per_w*2],
            x[...,per_w*2:per_w*3],
            x[...,per_w*3:per_w*4],
            x[...,per_w*4:per_w*5],
            x[...,per_w*5:]
        ]
        res = [hd_crop(_x) for _x in res]
        res = torch.cat(res,dim=-1)
        return res
    
    def crop_drivewm(x,in_shape):
        # 224x400 pad-top-> 225x400 fix-ratio-resize-> 216x384 top-crop-> 192x384
        resize=transforms.Resize((216,384))
        per_w=in_shape[2]//6
        res = [
            x[...,:per_w],
            x[...,per_w:per_w*2],
            x[...,per_w*2:per_w*3],
            x[...,per_w*3:per_w*4],
            x[...,per_w*4:per_w*5],
            x[...,per_w*5:]
        ]
        res = [hd_crop(resize(pad_top(_x)),(216,384),(192,384)) for _x in res]
        res = torch.cat(res,dim=-1)
        return res

    def get_occ_proj(occ_proj,cfg,use_occ_3d,use_occ_3d_fg,use_occ_3d_bg): # only pass in those difference between controlnets
        if occ_proj is not None and not use_occ_3d:
            if cfg.dataset.image_size in [[224,400],[432,768]]:
                # modify: occ in place of original 'bev mask' to passed to controlnet
                in_shape = occ_proj[examples[0]['metas'].data['token']].shape
                # assert in_shape==(3,224,400*6)
                ds_occ_proj = torch.stack([
                    occ_proj[example['metas'].data['token']] for example in examples], dim=0)
            elif cfg.dataset.image_size == [192,384]:
                # modify: occ in place of original 'bev mask' to passed to controlnet
                in_shape = occ_proj[examples[0]['metas'].data['token']].shape
                assert in_shape==(3,224,400*6)
                ds_occ_proj = torch.stack([
                    crop_drivewm(occ_proj[example['metas'].data['token']],in_shape) for example in examples], dim=0)
            else: # 256x704
                in_shape = occ_proj[examples[0]['metas'].data['token']].shape
                assert in_shape==(3,432,768*6)
                ds_occ_proj = torch.stack([
                    crop_hd(occ_proj[example['metas'].data['token']],in_shape) for example in examples], dim=0)
        elif occ_proj is not None and use_occ_3d:  # should not exist alongside occ_proj
            if cfg.dataset.image_size in [[224,400],[432,768]]:
                # modify: occ in place of original 'bev mask' to passed to controlnet
                tmp = [occ_proj.project(example['metas'].data['token']) for example in examples]
                if not use_occ_3d_fg:
                    tmp = [torch.where(_tmp<=10, 17, _tmp) for _tmp in tmp] # filter out foregrounds
                if not use_occ_3d_bg:
                    tmp = [torch.where(11<=_tmp, 17, _tmp) for _tmp in tmp] # filter out backgrounds
                ds_occ_proj = torch.stack(tmp, dim=0) # bs,n_cam,h,w,c
                ds_occ_proj = rearrange(ds_occ_proj, 'b n ... -> (b n) ...').permute(0,3,1,2)  # bs*n_cam,c,h,w
                ds_occ_proj = ds_occ_proj.float()  # float32
                ds_occ_proj /= 17  # normalize to 0~1
            else:
                raise NotImplementedError
        return ds_occ_proj
    if cfg.use_dual_controlnet:
        ds_occ_proj = []
        for i in range(len(occ_proj)):
            ds_occ_proj.append(get_occ_proj(occ_proj[i],cfg,cfg.use_occ_3d[i],cfg.use_occ_3d_fg[i],cfg.use_occ_3d_bg[i],))
    else:
        ds_occ_proj=get_occ_proj(occ_proj,cfg,cfg.use_occ_3d,cfg.use_occ_3d_fg,cfg.use_occ_3d_bg)

    # camera param
    # TODO: camera2lidar should be changed to lidar2camera
    # fmt: off
    camera_param = torch.stack([torch.cat([
        example["camera_intrinsics"].data[:, :3, :3],  # 3x3 is enough
        example["camera2lidar"].data[:, :3],  # only first 3 rows meaningful
    ], dim=-1) for example in examples], dim=0)
    # fmt: on

    ret_dict = {
        "bev_map_with_aux": bev_map_with_aux,
        "occ_proj": ds_occ_proj,
        "camera_param": camera_param,
        "kwargs": {},
    }

    if "img" in examples[0]:
        # multi-view images
        pixel_values = torch.stack(
            [example["img"].data for example in examples])
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()
        ret_dict["pixel_values"] = pixel_values
    elif is_train:
        raise RuntimeError("For training, you should provide gt images.")

    # bboxes_3d, convert to tensor
    # here we consider:
    # 1. do we need to filter bboxes for each view? use `view_shared`
    # 2. padding for one batch of data if need (with zero), and output mask.
    # 3. what is the expected output format? dict of kwargs to bbox embedder
    canvas_size = pixel_values.shape[-2:]
    def get_bboxes_3d(map_vec, use_map_vec, bbox_view_shared): # only pass in those difference between controlnets
        if bbox_mode is not None:
            if not use_map_vec: 
                # NOTE: both can be None
                bboxes_3d_input, bbox_view_coord = _preprocess_bbox(
                    bbox_mode, canvas_size, examples, is_train=is_train,
                    view_shared=bbox_view_shared, bbox_add_ratio=bbox_add_ratio,
                    bbox_add_num=bbox_add_num, bbox_drop_ratio=bbox_drop_ratio)
                # ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input
            else:
                boxes_m = []
                clses_m = []
                for example in examples:
                    box_m, cls_m = map_vec[example['metas'].data['token']]
                    boxes_m.append(box_m)
                    clses_m.append(cls_m)
                bboxes_3d_input = _preprocess_map_vec(cfg,boxes_m,clses_m,
                                                      view_shared=bbox_view_shared)
                # ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input
        else:
            bbox_view_coord = None
        return bboxes_3d_input
    if cfg.use_dual_controlnet:
        bboxes_3d_input = []
        for i in range(len(cfg.use_map_vec)):
            bboxes_3d_input.append(get_bboxes_3d(map_vec[i], cfg.use_map_vec[i], bbox_view_shared[i]))
    else:
        bboxes_3d_input = get_bboxes_3d(map_vec, cfg.use_map_vec, bbox_view_shared)
    ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input

    # get classes as 'sentences'
    txt_aug=None if not cfg.use_aug_text else []
    ret_dict["kwargs"]['use_aug_text'] = cfg.use_aug_text
    if bboxes_3d_input is not None and cfg.use_aug_text:
        obj_cls=list(cfg.dataset.object_classes)
        cls = ret_dict["kwargs"]["bboxes_3d_data"]["classes"]
        for bs in range(len(cls)):
            for cam in range(6):
                cur_cls = cls[bs][cam]
                # 1. for caption augmentation
                name,count = torch.unique(cur_cls,return_counts=True)
                name,count = name[1:],count[1:] # there is a -1 in the front as blank
                name = map(lambda x:obj_cls[x], name.tolist())
                txt_aug.append(', '.join(name))
                # # 2. for box alignment
                # mapped_cls_str = ', '.join(map(lambda x:obj_cls[x] if x!=-1 else -1, cur_cls.tolist()))

    # captions: one real caption with one null caption
    input_ids_padded, captions = _tokenize_captions(
        examples, template, tokenizer, is_train, txt_aug)
    ret_dict["captions"] = captions[:-1]  # list of str
    if tokenizer is not None:
        # real captions in head; the last one is null caption
        # we omit "attention_mask": padded_tokens.attention_mask, seems useless
        ret_dict["input_ids"] = input_ids_padded[:-1]
        ret_dict["uncond_ids"] = input_ids_padded[-1:]

    # other meta data
    meta_list_dict = dict()
    for key in META_KEY_LIST:
        try:
            meta_list = [example[key] for example in examples]
            meta_list_dict[key] = meta_list
        except KeyError:
            continue
    ret_dict['meta_data'] = meta_list_dict

    if is_train and cfg.use_aug_loss:
        # assert not cfg.use_map_vec
        # NOTE: view_shared hard code to False here
        for_mask, _ = _preprocess_bbox(
            bbox_mode, canvas_size, examples, is_train=is_train,
            view_shared=False, bbox_add_ratio=bbox_add_ratio,
            bbox_add_num=bbox_add_num, bbox_drop_ratio=bbox_drop_ratio,
            use_3d_filter=False, for_mask=True)
        lidar2image = [torch.bmm(example['camera_intrinsics'].data, example['lidar2camera'].data) for example in examples]
        if cfg.dataset.image_size == [224,400]:
            heat_map_28_50 = create_heatmap_gt(for_mask,lidar2image,meta_list_dict['metas'],resolution=(50,28)) # bs,n_cam,max_n_box,28,50
            # maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # heat_map_14_25 = maxpool(heat_map_28_50)
            # heat_map_7_13 = maxpool(heat_map_14_25)
            # heat_map_4_7 = maxpool(heat_map_7_13)
            ret_dict['heatmap_gt'] = [heat_map_28_50]#,heat_map_14_25,heat_map_7_13,heat_map_4_7,heat_map_4_7]
        elif cfg.dataset.image_size == [192,384]:
            # hidden size: 384//8 x 192//8 = 48 x 24
            # we should get original ratio 48x27 first, then crop top
            heat_map_27_48 = create_heatmap_gt(for_mask,lidar2image,meta_list_dict['metas'],resolution=(48,27)) # bs,n_cam,max_n_box,28,50
            heat_map_24_48 = hd_crop(heat_map_27_48,(27,48),(24,48))
            ret_dict['heatmap_gt'] = [heat_map_24_48]
        else:
            # hidden size: 704//8 x 256//8 = 88 x 32
            heat_map_54_96 = create_heatmap_gt(for_mask,lidar2image,meta_list_dict['metas'],resolution=(96,54))
            if cfg.dataset.image_size == [432,768]:
                ret_dict['heatmap_gt'] = [heat_map_54_96]
            elif cfg.dataset.image_size == [256,704]:
                heat_map_32_88 = hd_crop(heat_map_54_96,(54,96),(32,88))
                ret_dict['heatmap_gt'] = [heat_map_32_88]
        # visualize_heatmap(ret_dict,gt=True)
    return ret_dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def visualize_heatmap(ret_dict,gt=False):
    to_visualize = [i.detach().cpu() for i in ret_dict['heatmap_gt']] # list containing one (bs,n_cam,h,w)
    gt_img = [i.detach().cpu() for i in ret_dict['pixel_values']] # list containing bs (n_cam,3,h,w)
    # input heatmap: [n_level of (bs,n_cam,res0,res1)]
    
    gt='gt'if gt else 'pred'
    for bs in range(to_visualize[0].shape[0]):
        tokens=ret_dict['meta_data']['metas'][bs].data['token']
        images = []
        for level in range(len(to_visualize)):
            for cams in range(to_visualize[0].shape[1]):            
                image = to_visualize[level][bs][cams]
                res = image.shape[-2:]
                image = image.reshape(res[0],res[1],-1)
                image = image.numpy().astype(np.float)
                # image = image/image.max()
                # image = (image - image.min()) / (image.max() - image.min() +1e-6)
                cmap = plt.get_cmap('coolwarm')  # 选择颜色映射
                colored_image = np.zeros((res[0], res[1], 4), dtype=np.float)
                for j in range(res[0]):
                    for k in range(res[1]):
                        colored_image[j, k] = cmap(image[j, k])
                mapped_data_rgb = (colored_image[:, :, :3] * 255).astype(np.uint8)
                mapped_data_rgb = Image.fromarray(mapped_data_rgb)
                image = np.array(mapped_data_rgb.resize((mapped_data_rgb.size[0]*8, mapped_data_rgb.size[1]*8)))#(256, 256, 3)
                images.append(image)
        view_images(np.stack(images, axis=0),view=f'{tokens}_{cams}',savepath=f'./heatmap_{gt}_viz', save=True, num_rows=len(to_visualize))
        # visualize gt
        _gt_img = gt_img[bs]
        _gt_img = torch.cat([_gt_img[i] for i in range(len(_gt_img))], dim=-1)
        _gt_img = _gt_img*0.5+0.5
        transforms.ToPILImage()(_gt_img).save(os.path.join(f'./heatmap_{gt}_viz',f'{tokens}_rgb.png'))

def view_images(images, view, savepath, num_rows=1, offset_ratio=0.02, save=True):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    # initialize a ones matrix with the size of all pics joinning together
    # then fill in with the pics' pixel values
    # the title comes from text_under_image()
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                    w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if not os.path.exists(savepath):
        os.makedirs(savepath,exist_ok=True)
    if save:
        pil_img.save(os.path.join(savepath,f'{view}.png'))
