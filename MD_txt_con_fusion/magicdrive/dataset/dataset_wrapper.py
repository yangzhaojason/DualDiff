import os
from glob import glob

import torch
from mmcv.parallel.data_container import DataContainer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes, Box3DMode


class ListSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, dataset, list) -> None:
        self.dataset = dataset
        self.list = list

    def __getitem__(self, idx):
        return self.dataset[self.list[idx]]

    def __len__(self):
        return len(self.list)


class FolderSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, folder) -> None:
        self.dataset = glob(os.path.join(folder, "*.pth"))

    def __getitem__(self, idx):
        data = torch.load(self.dataset[idx])
        mmdet3d_format = {}
        mmdet3d_format['gt_masks_bev'] = data['gt_masks_bev']
        # fmt: off
        # in DataContainer
        mmdet3d_format['img'] = DataContainer(data['img'])
        mmdet3d_format['gt_labels_3d'] = DataContainer(data['gt_labels_3d'])
        mmdet3d_format['camera_intrinsics'] = DataContainer(data['camera_intrinsics'])
        mmdet3d_format['lidar2camera'] = DataContainer(data['lidar2camera'])
        mmdet3d_format['img_aug_matrix'] = DataContainer(data['img_aug_matrix'])
        mmdet3d_format['metas'] = DataContainer(data['metas'])
        # special class
        gt_bboxes_3d = data['gt_bboxes_3d'][:, :7]  # or all, either can work
        mmdet3d_format['gt_bboxes_3d'] = DataContainer(LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR))

        # recompute
        camera2lidar = torch.eye(4, dtype=data['lidar2camera'].dtype)
        camera2lidar = torch.stack([camera2lidar] * len(data['lidar2camera']))
        camera2lidar[:, :3, :3] = data['lidar2camera'][:, :3, :3].transpose(1, 2)
        camera2lidar[:, :3, 3:] = torch.bmm(-camera2lidar[:, :3, :3], data['lidar2camera'][:, :3, 3:])
        mmdet3d_format['camera2lidar'] = DataContainer(camera2lidar)
        mmdet3d_format['lidar2image'] = DataContainer(
            torch.bmm(data['camera_intrinsics'], data['lidar2camera'])
        )
        # fmt: on
        return mmdet3d_format

    def __len__(self):
        return len(self.dataset)
    
    
import pickle
from PIL import Image
from torchvision import transforms
class OccFolderSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, folder, mode='image') -> None:
        if mode=='feature':
            self.dataset = glob(os.path.join(folder, "*.pkl"))
        elif mode=='image':
            self.totensor= transforms.ToTensor()
            self.dataset = glob(os.path.join(folder, "*.png"))
        self.root = folder
        self.mode = mode

    def __getitem__(self, token):
        if self.mode == 'feature':
            idx = self.dataset.index(os.path.join(self.root,f'{token}.pkl'))
            with open(self.dataset[idx], 'rb') as f:
                pkl_dict = pickle.load(f)
            return list(pkl_dict.values())[0]
        elif self.mode == 'image':
            idx = self.dataset.index(os.path.join(self.root,f'{token}.png'))
            img = Image.open(self.dataset[idx]).convert("RGB")#RGBA->RGB
            return self.totensor(img)
        # return torch.rand(3,432,768*6)

    def __len__(self):
        return len(self.list)


import json
class VecMapFolderSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, train,val) -> None:
        file1 = json.load(open(train,'rb'))
        file2 = json.load(open(val,'rb'))
        self.dataset = {}
        for item in file1['GTs']:
            self.dataset[item['sample_token']]=item['vectors']
        for item in file2['GTs']:
            self.dataset[item['sample_token']]=item['vectors']

    def __getitem__(self, token):
        # import random;token=random.choice(['53f5977684e14cb0a28f383fee1dd433','a9460d1306f94aaf894af8b07d5f23a9'])
        data = self.dataset[token]
        if not data: # empty map in this sample
            return None, None
        gt_bboxes_3d = torch.stack([torch.tensor(i['pts_fixed_num']) for i in data]) #n,8,2
        gt_bboxes_3d_z = torch.zeros(*gt_bboxes_3d.shape[:-1],1)
        gt_bboxes_3d = torch.cat([gt_bboxes_3d,gt_bboxes_3d_z],dim=-1)  # NOTE: mnually add z coord(as 0) to all
        gt_labels_3d = torch.stack([torch.tensor(i['type']) for i in data]) #n
        return gt_bboxes_3d, gt_labels_3d

    def __len__(self):
        return len(self.dataset)