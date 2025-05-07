import os
import sys
import math
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
sys.path.append(".")  # noqa
from typing import List
from PIL import Image,ImageOps
import numpy as np
import cv2
from tqdm import trange

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pickle
# from nuscenes.nuscenes import NuScenes

from magicdrive.dataset import FolderSetWrapper,OccFolderSetWrapper
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
    UNet2DConditionModel,)
from packaging import version

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on


def run_inference(rank, world_size, model_path, savepath, image_size, overrides, batchsize=3):
    # val_dataset,val_idx,pipeline,nusc,cfg=main_params['val_dataset'],main_params['val_idx'],main_params['pipeline'],main_params['nusc'],main_params['cfg']
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    weight_dtype = torch.float32

    # dumb workaround, otherwise don't know how to get HydraConfig.get()
    # @hydra.main(version_base=None, config_path="../configs", config_name="my_test_config")
    # def get_cfg(cfg: DictConfig):
    #     return HydraConfig.get().runtime.output_dir
    # load config
    initialize(config_path="../configs", version_base=None)
    cfg = compose(config_name="config", overrides=[f'+exp={overrides}'])
    # setattr(cfg, "log_root", get_cfg())
    print('='*9,'config loaded','='*9)

    # TODO: only workaround cuz original code use accelerator to load trained state dict
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
    #     mixed_precision=cfg.accelerator.mixed_precision,
    #     log_with=cfg.accelerator.report_to,
    #     project_dir=cfg.log_root,
    #     kwargs_handlers=[ddp_kwargs],
    # )

    # model
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")

    model_cls = load_module(cfg.model.model_module)
    controlnet_param = OmegaConf.to_container(cfg.model.controlnet, resolve=True)
    controlnet = model_cls.from_pretrained(os.path.join(model_path,'controlnet'),torch_dtype=weight_dtype,low_cpu_mem_usage=False,device_map=None)
    
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
    unet = unet_cls.from_pretrained(os.path.join(model_path,'unet'), torch_dtype=weight_dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.eval()
    controlnet.eval()

    if is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    else:
        # raise ValueError("xformers is not available. Make sure it is installed correctly")
        pass
    print('='*9,'model loaded','='*9)


    # dataset
    import math
    # val_dataset = FolderSetWrapper('/opt/data/private/aigc/MagicDrive_outdated/train/data/val_occ')
    val_dataset = build_dataset(
        OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
    )
    # val_idx = cfg.runner.validation_index if len(cfg.runner.validation_index) != 0 else [i for i in range(len(val_dataset))]
    val_idx = [i for i in range(len(val_dataset))]
    shard_volumn = math.ceil(len(val_idx) / world_size)
    # nusc = NuScenes(dataroot=cfg.dataset.dataset_root,version='v1.0-trainval',verbose=False) # do this outside run_inference() would cause SIGKILL at mp.spawn
    nusc = {}
    nusc.update(pickle.load(open('filenames_0.pkl','rb')))
    nusc.update(pickle.load(open('filenames_1.pkl','rb')))
    nusc.update(pickle.load(open('filenames_2.pkl','rb')))
    # savepath = cfg.savepath
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
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.set_progress_bar_config(disable=True)
    print('='*9,'pipeline loaded','='*9)
    

    pipeline.to(rank)
    torch.cuda.set_device(rank)
    # ds_per_GPU = split_dataset_by_node(ds, rank=rank, world_size=world_size)
    # nusc = NuScenes(dataroot=cfg.dataset.dataset_root,version='v1.0-trainval',verbose=False)
    idx_per_GPU = val_idx[rank*shard_volumn: (rank+1)*shard_volumn if ((rank+1)*shard_volumn < len(val_idx)) else len(val_idx)]
    batch_num = math.ceil(len(idx_per_GPU) / batchsize)
    for validation_i in trange(batch_num):
        # conditions = []
        # image_file_name,prompt = ds_per_GPU[i]['sample_data_info']['filename'],ds_per_GPU[i]['prompt']
        # for img_modal in cnet_inputs:
        #     conditions.append(ds_per_GPU[i][img_modal].resize(eval(args.resolution)))
        batched_raw_data = []
        for _i in range(batchsize):
            if validation_i+_i > (len(idx_per_GPU)-1): break
            raw_data = val_dataset[idx_per_GPU[validation_i+_i]]  # cannot index loader
            batched_raw_data.append(raw_data)
        val_input = collate_fn(
            batched_raw_data, cfg.dataset.template, is_train=False,
            bbox_mode=cfg.model.bbox_mode,
            bbox_view_shared=cfg.model.bbox_view_shared,
            occ_proj=OccFolderSetWrapper('./occ_proj/occ_bg',mode='image'),
            cfg=cfg,)
        
        count=0
        for _i in range(len(batched_raw_data)):
            # sample_ = nusc.get('sample',val_input['meta_data']['metas'][_i].data['token'])
            filenames = nusc[val_input['meta_data']['metas'][_i].data['token']]
            for view_i in range(len(cfg.dataset.view_order)):
                # sample_data_ = nusc.get('sample_data',sample_['data'][cfg.dataset.view_order[view_i]])
                # filename = sample_data_['filename']
                filename = filenames[view_i]
                assert cfg.dataset.view_order[view_i] == filename.split('/')[1]
                if os.path.exists(os.path.join(savepath, filename)):
                    count+=1
        if count == 6*len(batched_raw_data): continue

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
        with torch.autocast("cuda"):
            image: BEVStableDiffusionPipelineOutput = pipeline(
                prompt=val_input["captions"],
                image=val_input["occ_proj"],
                camera_param=camera_param,
                height=cfg.dataset.image_size[0],
                width=cfg.dataset.image_size[1],
                generator=generator,
                bev_controlnet_kwargs=val_input["kwargs"],
                **cfg.runner.pipeline_param,
            )
            # assert len(image.images) == 1
            images: List[List[Image.Image]] = image.images
            # image: List[Image.Image] = image.images[0]
        # gen_list = image

        assert len(images) == len(batched_raw_data)
        for _i in range(len(batched_raw_data)):
            gen_list = images[_i]
            # sample_ = nusc.get('sample',val_input['meta_data']['metas'][_i].data['token'])
            filenames = nusc[val_input['meta_data']['metas'][_i].data['token']]
            for view_i in range(len(cfg.dataset.view_order)):
                # sample_data_ = nusc.get('sample_data',sample_['data'][cfg.dataset.view_order[view_i]])
                # filename = sample_data_['filename']
                filename = filenames[view_i]
                assert cfg.dataset.view_order[view_i] == filename.split('/')[1]
                # TODO: simply resize to given size is not proper
                """
                # Unet in controlnet is adaptive to input size, that means if the input condition image is a bit
                # larger(as well as the sample noise, they are of the same shape), then all of the feature shape
                # in the unet flow would all be accordingly larger, and in the case the mergence of sd&controlnet
                # would not come across any issue. While here in magicdrive, the controlnet input is a fixed size
                # occ feature, so the controlnet output feature wouldn't align with that of the sd if the height
                # and width of the sample noise is changed. need to modify the feature of the controlnet or ...
                """
                """
                another detail: for original controlnet, during training, sd has a vae in the front, while control
                net has ControlNetConditioningEmbedding in the front. During inference(pipeline), both have vae in
                the front.
                """
                img_ = gen_list[view_i].resize(image_size[::-1])
                img_ = ImageOps.pad(img_,(1600,900),method=Image.BICUBIC,color=(0),centering=(0.5,1))
                try:
                    img_.save(os.path.join(savepath, filename))
                except FileNotFoundError:
                    tmp=len(filename.split('/')[-1])
                    path_to_create = os.path.join(savepath, filename[:-tmp])
                    os.system(f'mkdir -p {path_to_create}')
                    img_.save(os.path.join(savepath, filename))
def run_inference_debug(rank, world_size):
    print(rank,world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # ds_per_GPU = split_dataset_by_node(ds, rank=rank, world_size=world_size)
    ds_per_GPU = val_dataset[rank*shard_volumn: (rank+1)*shard_volumn if ((rank+1)*shard_volumn < len(val_dataset)) else len(val_dataset)]
    raw_data = ds_per_GPU[0]
    val_input = collate_fn(
        [raw_data], cfg.dataset.template, is_train=False,
        bbox_mode=cfg.model.bbox_mode,
        bbox_view_shared=cfg.model.bbox_view_shared,
    )
    print(val_input['meta_data']['metas'][0].data['token'])
    
def main():
    model_path = input('input model path(e.g. magicdrive-log/64000-ckpt): ')#'magicdrive-log/64000-ckpt'
    
    overrides = model_path.split('/')[-1].split('-')[1]
    if input(f'use overrides {overrides}? y/n: ') != 'y':
        overrides = input('input override(e.g. occ_bg): ')
    overrides = '224x400' if overrides is None else overrides
    
    savepath = 'downstream-'+model_path.split('/')[-1].split('-')[0]
    if input(f'use savepath {savepath}? y/n: ') != 'y':
        savepath = input('input save path(e.g. downstream-64000): ')#'downstream-64000'
    savepath = savepath+'_'+overrides

    batchsize = 3
    if input(f'use batchsize {batchsize}? y/n: ') != 'y':
        batchsize = eval(input('input batchsize(e.g. 3): '))

    image_size = [224,400]
    world_size = eval(input('input num of gpus: '))#cfg.GPU_num
    
    mp.spawn(run_inference, args=(world_size,model_path,savepath,image_size,overrides,batchsize), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()