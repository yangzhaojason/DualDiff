import os
import sys
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
sys.path.append(".")  # noqa
from typing import List
from PIL import Image
from tqdm import trange

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from nuscenes.nuscenes import NuScenes

from magicdrive.dataset import FolderSetWrapper
from magicdrive.misc.common import load_module
from magicdrive.pipeline.pipeline_bev_controlnet import (
    BEVStableDiffusionPipelineOutput,)
from magicdrive.dataset.utils import collate_fn
from mmdet3d.datasets import build_dataset

from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    UNet2DConditionModel,)
from packaging import version

from unet_modify import prep_unet, prep_controlnet

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

from magicdrive.dataset.dataset_wrapper import OccFolderSetWrapper
occ_proj = OccFolderSetWrapper('./occ_proj/occ_bg',mode='image')

def main(ip=''):
    weight_dtype = torch.float32

    # dumb workaround, otherwise don't know how to get HydraConfig.get()
    # @hydra.main(version_base=None, config_path="../configs", config_name="my_test_config")
    # def get_cfg(cfg: DictConfig):
    #     return HydraConfig.get().runtime.output_dir
    # load config
    initialize(config_path="../configs", version_base=None)
    cfg = compose(config_name="explore_config")
    # setattr(cfg, "log_root", get_cfg())
    print('='*9,'config loaded','='*9)

    # model
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
    # noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")

    model_cls = load_module(cfg.model.model_module)
    controlnet_param = OmegaConf.to_container(cfg.model.controlnet, resolve=True)
    # controlnet = model_cls.from_pretrained(os.path.join(cfg.official_ckpt,'controlnet'),torch_dtype=weight_dtype)
    controlnet = model_cls.from_pretrained(os.path.join(cfg.model_save_path,'controlnet'),torch_dtype=weight_dtype)
    
    '''since we don't use from_unet here, we should manually set some modules'''
    controlnet.use_cam_in_temb = cfg.model.controlnet.use_cam_in_temb
    if not controlnet.use_cam_in_temb:
        controlnet.adm_proj = None
    controlnet.use_txt_con_fusion = cfg.model.controlnet.use_txt_con_fusion
    if not controlnet.use_txt_con_fusion:
        controlnet.txt_con_fusion=None
    controlnet.use_txt_con_fusionp = cfg.model.controlnet.use_txt_con_fusionp
    if not controlnet.use_txt_con_fusionp:
        controlnet.txt_con_fusionp=None

    unet_cls = load_module(cfg.model.unet_module)
    unet = unet_cls.from_pretrained(os.path.join(cfg.model_save_path,'unet'), torch_dtype=weight_dtype)

    # controlnet = prep_controlnet(controlnet) # adapter controlnet
    controlnet = prep_unet(controlnet) # original controlnet
    unet = prep_unet(unet)

    # # load adapter controlnet after modifying layers
    # controlnet.load_state_dict(torch.load(os.path.join(cfg.model_save_path,'controlnet/diffusion_pytorch_model.bin')))

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.eval()
    controlnet.eval()

    # if is_xformers_available():
    #     import xformers
    #     xformers_version = version.parse(xformers.__version__)
    #     unet.enable_xformers_memory_efficient_attention()
    #     controlnet.enable_xformers_memory_efficient_attention()
    # else:
    #     # raise ValueError("xformers is not available. Make sure it is installed correctly")
    #     pass
    print('='*9,'model loaded','='*9)

    # dataset
    import math
    # val_dataset = FolderSetWrapper('/opt/data/private/aigc/MagicDrive_outdated/train/data/val_occ')
    val_dataset = build_dataset(OmegaConf.to_container(cfg.dataset.data.val, resolve=True))
    '''to visualize the 18 channels in gt_masks_bev
    if rank == 0:
        import numpy as np
        a = val_dataset[0]
        for i in range(len(a['gt_masks_bev'])):
            # Image.fromarray(a['gt_masks_bev'][i].astype(np.uint8)*255).show()
            Image.fromarray(a['gt_masks_bev'][i].astype(np.uint8)*255).save(f'viz_bev_map/{i}.png')
        os._exit(0)
    '''
    # if not len(cfg.runner.validation_index) == 0:  # if 0, go with full set
    #     val_dataset = [val_dataset[i] for i in cfg.runner.validation_index]
    # else:
    #     val_dataset = [val_dataset[i] for i in range(len(val_dataset))] # full set, too slow
    val_idx = cfg.runner.validation_index if len(cfg.runner.validation_index) != 0 else [i for i in range(len(val_dataset))]
    # shard_volumn = math.ceil(len(val_idx) / cfg.GPU_num)
    # nusc = NuScenes(dataroot=cfg.dataset.dataset_root,version='v1.0-trainval',verbose=False) # do this outside run_inference() would cause SIGKILL at mp.spawn
    savepath = cfg.downstream_savepath
    print('='*9,'dataset loaded','='*9)

    # pipeline
    pipe_param={
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }
    pipe_cls = load_module(cfg.model.pipe_module)
    pipeline = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        feature_extractor=None,  # since v1.5 has default, we need to override
        torch_dtype=weight_dtype,
    )
    # NOTE: this scheduler does not take generator as kwargs.
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config)

    # NOTE: test whether changing scheduler and also the inf steps(magicdrive vs drive-WM -> ddpm&20 vs ddim&50)
    # print(cfg.runner.pipeline_param.num_inference_steps,cfg.seed)
    cfg.runner.pipeline_param.num_inference_steps = 20;#cfg.seed=None
    # print(cfg.runner.pipeline_param.num_inference_steps,cfg.seed)
    # pipeline.scheduler=DDIMScheduler.from_config(pipeline.scheduler.config,clip_sample=False)
    # # print(pipeline.scheduler.compatibles)
    
    # pipeline.enable_xformers_memory_efficient_attention()
    pipeline.set_progress_bar_config(disable=True)
    print('='*9,'pipeline loaded','='*9)

    from mmcv.parallel.data_container import DataContainer
    from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes, Box3DMode
    
    pipeline.to('cuda')
    # torch.cuda.set_device(rank)
    # nusc = NuScenes(dataroot=cfg.dataset.dataset_root,version='v1.0-trainval',verbose=False)
    # idx_per_GPU = val_idx[rank*shard_volumn: (rank+1)*shard_volumn if ((rank+1)*shard_volumn < len(val_idx)) else len(val_idx)]
    idx_per_GPU = val_idx
    cnt = 0
    for validation_i in trange(len(idx_per_GPU)):
        # conditions = []
        # image_file_name,prompt = ds_per_GPU[i]['sample_data_info']['filename'],ds_per_GPU[i]['prompt']
        # for img_modal in cnet_inputs:
        #     conditions.append(ds_per_GPU[i][img_modal].resize(eval(args.resolution)))
        raw_data = val_dataset[idx_per_GPU[validation_i]]  # cannot index loader

        # # NOTE:test: would edit in box affect gen?(only one box)
        # # print(raw_data['gt_bboxes_3d'],raw_data['gt_labels_3d'])
        # tmp = torch.tensor([1.5e+00,  5.6442e+00, -2.3853e-01,  2.0230e+00,  4.8720e+00,
        #   1.8150e+00,  9.6819e-02]).unsqueeze(0)  # x=0,y=2.6m, which means in the front
        # raw_data['gt_bboxes_3d'] = DataContainer(LiDARInstance3DBoxes(
        #         tmp, box_dim=tmp.shape[-1],
        #         origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR))
        # raw_data['gt_labels_3d'] = DataContainer(torch.tensor([0])) # cls 'car'

        val_input = collate_fn(
            [raw_data], cfg.dataset.template, is_train=False,
            bbox_mode=cfg.model.bbox_mode,
            bbox_view_shared=cfg.model.bbox_view_shared,
            occ_proj=occ_proj, cfg=cfg
        )
        # camera_emb = self._embed_camera(val_input["camera_param"])
        camera_param = val_input["camera_param"].to(weight_dtype)

        # let different prompts have the same random seed
        if cfg.seed is None:
            generator = None
        else:
            generator = torch.Generator(device="cuda").manual_seed(
                cfg.seed
            )

        # for each input param, we generate several times to check variance.
        gen_list, gen_wb_list = [], []
        # for _ in range(self.cfg.runner.validation_times):
        # only generate one time for each sample
        
        # NOTE:test: would edit in prompt affect gen?(turns out no)
        # val_input["captions"][0] = val_input["captions"][0] + ' At night.'

        with torch.autocast("cuda"):
            image: BEVStableDiffusionPipelineOutput = pipeline(
                prompt=val_input["captions"],
                image=val_input["bev_map_with_aux"], #val_input["occ_proj"], 
                camera_param=camera_param,
                height=cfg.dataset.image_size[0],
                width=cfg.dataset.image_size[1],
                generator=generator,
                bev_controlnet_kwargs=val_input["kwargs"],
                **cfg.runner.pipeline_param,
            )
            assert len(image.images) == 1
            # image: List[Image.Image] = image.images[0]
            cross_attn_map = image.cross_attn_map
            gen_list = image.images[0]

        caption = val_input["captions"][0]
        # cross_attn_map['caption'] = caption
        token = val_input['meta_data']['metas'][0].data['token']
        save_ = os.path.join(cfg.log_root,'explore',token,ip)
        os.makedirs(save_, exist_ok=True)
        torch.save(cross_attn_map, os.path.join(save_,f'{token}_attn.pth'))
        with open(os.path.join(save_,f'{token}_attn.txt'), 'w') as f:
            f.write(caption)
            # f.writelines(val_input['meta_data']['metas'][0].data)
        # sample_ = nusc.get('sample',val_input['meta_data']['metas'][0].data['token'])
        for view_i in range(len(cfg.dataset.view_order)):
            # sample_data_ = nusc.get('sample_data',sample_['data'][cfg.dataset.view_order[view_i]])
            # filename = sample_data_['filename']
            # # TODO: simply resize to given size is not proper
            # """
            # # Unet in controlnet is adaptive to input size, that means if the input condition image is a bit
            # # larger(as well as the sample noise, they are of the same shape), then all of the feature shape
            # # in the unet flow would all be accordingly larger, and in the case the mergence of sd&controlnet
            # # would not come across any issue. While here in magicdrive, the controlnet input is a fixed size
            # # occ feature, so the controlnet output feature wouldn't align with that of the sd if the height
            # # and width of the sample noise is changed. need to modify the feature of the controlnet or ...
            # """
            # """
            # another detail: for original controlnet, during training, sd has a vae in the front, while control
            # net has ControlNetConditioningEmbedding in the front. During inference(pipeline), both have vae in
            # the front.
            # """
            img_ = gen_list[view_i]#.resize(cfg.image_size[::-1])
            # try:
            img_.save(os.path.join(save_,f'./{view_i}.png'))
            # except FileNotFoundError:
            #     tmp=len(filename.split('/')[-1])
            #     path_to_create = os.path.join(savepath, filename[:-tmp])
            #     os.system(f'mkdir -p {path_to_create}')
            #     img_.save(os.path.join(savepath, filename))
        # cnt += 1
        # if cnt >1:
        #     break

        
    # def main():
    #     world_size = 4#cfg.GPU_num
    #     mp.spawn(run_inference, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main(ip='crop')