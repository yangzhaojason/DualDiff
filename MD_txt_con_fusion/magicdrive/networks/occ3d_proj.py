import torch
import torch.nn.functional as F
import time
import os
import copy
import numpy as np
import cv2
import pickle
from pyquaternion import Quaternion
class OccupancyRay:
    def __init__(self,image_shape=(900,1600),sample_point=200,sample_step=0.2,compress_ratio=8,dataroot='./data/nuscenes/',device='cpu'):
        pkl_root='magicdrive/networks'
        with open(os.path.join(pkl_root,'camera.pkl'), 'rb') as f:
            self.camera_data = pickle.load(f)
        with open(os.path.join(pkl_root,'occ3d_idx.pkl'), 'rb') as f:
            self.occ3d_idx = pickle.load(f)
        self.device=device
        self.dataroot=dataroot
        self.image_shape=image_shape
        self.sample_point=sample_point
        self.sample_step=sample_step
        self.compress_ratio=compress_ratio
        self.image_shape_compress=[int(self.image_shape[0]*compress_ratio),
                                   int(self.image_shape[1]*compress_ratio)]
    def compute_rays(self,K, Rt, u_array, v_array):
        u_array=u_array.to(torch.float32)
        v_array=v_array.to(torch.float32)
        K=K.to(torch.float32)
        Rt=Rt.to(torch.float32)
        assert len(u_array) == len(v_array), "u_array and v_array must have the same length"
        
        K_inv = torch.inverse(K)
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        ones = torch.ones_like(u_array)
        pixels_homogeneous = torch.stack([u_array, v_array, ones], dim=1).to(self.device)
        p_c = torch.matmul(K_inv, pixels_homogeneous.T).T
        d = torch.matmul(R, p_c.T).T
        d = d / torch.norm(d, dim=1, keepdim=True)
        ray_origins = t.expand_as(d)
        return ray_origins, d
    """
    CAM_FRONT: /opt/data/private/aigc/vidar/ViDAR/data/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg
    CAM_FRONT_RIGHT: /opt/data/private/aigc/vidar/ViDAR/data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281439770339.jpg
    CAM_BACK_RIGHT: /opt/data/private/aigc/vidar/ViDAR/data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281439777893.jpg
    CAM_BACK: /opt/data/private/aigc/vidar/ViDAR/data/nuscenes/samples/CAM_BACK/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281439787525.jpg
    CAM_BACK_LEFT: /opt/data/private/aigc/vidar/ViDAR/data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281439797423.jpg
    CAM_FRONT_LEFT: /opt/data/private/aigc/vidar/ViDAR/data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg
    """
    def project(self,sample_token):
        occ_gt_file = os.path.join(self.dataroot,self.occ3d_idx[sample_token],'labels.npz')
        occ_gt = np.load(occ_gt_file)['semantics'] # 200x200x16
        occ_gt = torch.tensor(occ_gt).to(self.device).to(torch.int64).unsqueeze(0) # 1x200x200x16
        # t0=time.time()
        # print(occ_feat.shape)
        # occ_feat=torch.argmax(occ_feat,dim=-1).unsqueeze(-1)
        # print(occ_feat.shape)
        one_hot_tensor =  F.one_hot(occ_gt, num_classes=18).float()
        occ_feat = one_hot_tensor.permute([0,4, 1, 2, 3]).contiguous()  # 原始形状即为 (1, 200, 200, 16, 18)
        
        cam_data=self.camera_data[sample_token]
        outputs=[]
        pts=[]

        for key in ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT']:
            # Step 1. Get Camera Intrinsic and Extrinsic
            # print(key)
            translation=cam_data[key]['translation']
            rotation=cam_data[key]['rotation']
            translation = np.array(translation)
            rotation = np.array(rotation)
            rotation_matrix = Quaternion(rotation).rotation_matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation
            Rt=torch.from_numpy(transformation_matrix).to(self.device).to(torch.float32) # 外参矩阵
            K=torch.Tensor(cam_data[key]['intrinsic']).to(self.device).to(torch.float32) # 内参矩阵
            x_range = torch.arange(0, self.image_shape_compress[1]).to(self.device)
            y_range = torch.arange(0, self.image_shape_compress[0]).to(self.device)
            xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
            grid_x = xx.flatten()//self.compress_ratio
            grid_y = yy.flatten()//self.compress_ratio
            # print(K)
            # x+ 是左  y+ 是前
            ray_origin,d=self.compute_rays(K,Rt,grid_x,grid_y)
            ray_origin=ray_origin.view(self.image_shape_compress[1],self.image_shape_compress[0],3).permute(1,0,2).contiguous()
            d=d.view(self.image_shape_compress[1],self.image_shape_compress[0],3).permute(1,0,2).contiguous()
            # d[...,0],d[...,1]=d[...,1],d[...,0]
            # print(ray_origin.shape,d.shape)
            steps = torch.arange(self.sample_point).float().to(self.device)*self.sample_step # +-40m
            points = ray_origin.unsqueeze(2) + steps.view(1, 1, -1, 1) * d.unsqueeze(2) # 单位 : m
            # print(points.shape) # 116 200 200 3
            grid=points /40 
            grid[...,2]=grid[...,2]*40/3.2 -2.2/3.2 # -1 ~ +5.4
            grid_copy = grid.clone()  
            grid[..., 1], grid[..., 2],grid[...,0] = grid_copy[..., 1], grid_copy[..., 0] ,grid_copy[...,2]
            # print("grid shape",grid.shape)# 116 200 200 3
            grid = grid.view(1, self.image_shape_compress[0]*(self.image_shape_compress[1])*self.sample_point, 1, 1, 3).contiguous()
            
            
            # pts.append(grid.cpu())
            output = F.grid_sample(occ_feat, grid, mode='nearest', padding_mode='zeros', align_corners=False)
            output = output.squeeze(-1).squeeze(-1).view(1,18,self.image_shape_compress[0],
            self.image_shape_compress[1],self.sample_point).permute(0,2,3,4,1).contiguous()
            idx=torch.where(torch.sum(output,dim=-1)==0) # because zeros padding mode.
            output[idx[0],idx[1],idx[2],idx[3],17]=1
            
            output = torch.argmax(output,dim=-1) # to 200 200 16 value:0~17
            outputs.append(output.squeeze(0))
            # 输出最终形状
            # print(output.shape)
        # print(time.time()-t0)
        return torch.stack(outputs,dim=0)   # 6, h//8, w//8, 200
        # result={"outputs":outputs,"token":sample_token,"occ_gt":occ_feat.cpu(),"points":pts}
        # with open("/opt/data/private/aigc/vidar/occ_data.pkl", 'wb') as file:
        #     pickle.dump(result, file)

def get_img(nusc, sample_token, nuscroot='/opt/data/private/nuscenes', w=1600, h=900):
    sample = nusc.get('sample',sample_token)
    imgs_tensor = []
    for view in ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                'CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT']:
        sample_data_item = nusc.get('sample_data',sample['data'][view])
        img = Image.open(os.path.join(nuscroot,sample_data_item['filename']))
        # img.save('./delete.png')
        img2tensor = np.asarray(img.resize((w,h)))
        img2tensor = torch.tensor(img2tensor) / 255
        img2tensor = img2tensor.permute(2,0,1)
        img2tensor = img2tensor*2 - 1  # 0~1 -> -1~1
        imgs_tensor.append(img2tensor)
    return torch.stack(imgs_tensor)

if __name__ == '__main__':
    '''
    some nuscenes mini sample tokens for you to check:
    'ca9a282c9e77460f8360f564131a8af5'  # mini sample 0
    'ac452a60e8b34a7080c938c904b23057'  # mini sample 100
    'f3e7e088082d4aabb9a3bf504d4ac904'  # mini sample 200
    '7f87e737dad642d9aff78a7fe12ce194'  # mini sample 300
    'd7cb9aa06de1442d8e2a22d562045cb4'  # mini sample 400
    '''
    from nuscenes import NuScenes
    from PIL import Image
    h = 896; w = 1600  # NOTE: SHOULD NOT BE CHANGED, otherwise cam intrinsics should also be changed
    sample_point = 320
    compress_ratio= 768 / 8 / 1600
    projector = OccupancyRay(image_shape=(h,w),sample_point=sample_point,compress_ratio=compress_ratio) # output h = h_in // compress_ratio
    nusc = NuScenes(version='v1.0-mini',dataroot='/opt/data/private/nuscenes', verbose=False)
    sample_token = 'd7cb9aa06de1442d8e2a22d562045cb4'
    # sample_token = 'c6d18bda94224330b898c3c2ff32fa74'
    # sample_token = '603c9225a5624229bae41a819d10c9cd'
    img = get_img(nusc, sample_token, w=int(w*compress_ratio), h=int(h*compress_ratio))  # 6,3,h,w
    project_feat = projector.project(sample_token) # 6,int(h*compress_ratio),int(w*compress_ratio),sample_point
    print(project_feat.shape)

    # filter out foregrounds
    # project_feat = torch.where(project_feat<=10, 17, project_feat)

    # filter out backgrounds
    # project_feat = torch.where(11<=project_feat, 17, project_feat)

   
    # print(img[0].shape)
    downsampled_images = np.transpose(img.numpy(), (0, 2, 3, 1)) # 6,h,w,3
    reshaped_tensor = project_feat.view(-1, sample_point)
    non_17_mask = reshaped_tensor != 17
    first_non_17_indices = non_17_mask.float().argmax(dim=1)
    first_non_17_values = reshaped_tensor[torch.arange(reshaped_tensor.size(0)), first_non_17_indices]
    result_occ = first_non_17_values.view(6, int(h*compress_ratio), int(w*compress_ratio))
    numpy_array=result_occ.numpy()
    colors_map = np.array(
    [
        [  0,   0,   0, 255],       #  0 undefined identities 
        [255, 120,  50, 255],       #  1 barrier              orange
        [255, 192, 203, 255],       #  2 bicycle              pink
        [255, 255,   0, 255],       #  3 bus                  yellow
        [  0, 150, 245, 255],       #  4 car                  blue
        [  0, 255, 255, 255],       #  5 construction_vehicle cyan
        [255, 127,   0, 255],       #  6 motorcycle           dark orange
        [255,   0,   0, 255],       #  7 pedestrian           red
        [255, 240, 150, 255],       #  8 traffic_cone         light yellow
        [135,  60,   0, 255],       #  9 trailer              brown
        [160,  32, 240, 255],       # 10 truck                purple                
        [255,   0, 255, 255],       # 11 driveable_surface    dark pink
        [139, 137, 137, 255],       # 12 other_flat           dark red
        [ 75,   0,  75, 255],       # 13 sidewalk             dard purple
        [150, 240,  80, 255],       # 14 terrain              light green          
        [230, 230, 250, 255],       # 15 manmade              white
        [  0, 175,   0, 255],       # 16 vegetation           green
        [  0,   0,   0, 255],       # 17 not occupied
    ])[:,:3]
    colored_images = np.zeros((6, int(h*compress_ratio), int(w*compress_ratio), 3), dtype=np.uint8)
    for i in range(18):
        colored_images[numpy_array == i] = colors_map[i]
    # print(colored_images.shape,downsampled_images.shape)
    downsampled_images=(downsampled_images-np.min(downsampled_images))/(np.max(downsampled_images)-np.min(downsampled_images))*255
    colored_image=torch.tensor(np.concatenate([colored_images[i] for i in range(len(colored_images))],axis=-2))
    downsampled_image=torch.tensor(np.concatenate([downsampled_images[i] for i in range(len(downsampled_images))],axis=-2))
    mask_fg = torch.tensor(colored_image == 0)
    mask_bg = torch.tensor(colored_image != 0)
    res=colored_image.masked_fill(mask_fg,0)*0.7 \
        +downsampled_image.masked_fill(mask_fg,0)*0.3 \
        +downsampled_image.masked_fill(mask_bg,0)
    
    print(res.numpy().shape)
    cv2.imwrite(f"./delete.png",cv2.cvtColor(res.numpy().astype(np.uint8),cv2.COLOR_RGB2BGR))
    
