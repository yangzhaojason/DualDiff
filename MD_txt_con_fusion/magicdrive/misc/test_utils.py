from typing import Tuple, Union, List
import os
import logging
from hydra.core.hydra_config import HydraConfig
from functools import partial
from omegaconf import OmegaConf
from omegaconf import DictConfig,ListConfig
from PIL import Image

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from mmdet3d.datasets import build_dataset
from diffusers import UniPCMultistepScheduler
import accelerate
from accelerate.utils import set_seed

from magicdrive.dataset import collate_fn, ListSetWrapper, FolderSetWrapper
from magicdrive.pipeline.pipeline_bev_controlnet import (
    StableDiffusionBEVControlNetPipeline,
    BEVStableDiffusionPipelineOutput,
)
from magicdrive.runner.utils import (
    visualize_map, img_m11_to_01, show_box_on_views
)
from magicdrive.misc.common import load_module
from magicdrive.networks.unet_addon_rawbox import get_submodel_ckpt
from magicdrive.dataset.dataset_wrapper import OccFolderSetWrapper,VecMapFolderSetWrapper
from magicdrive.networks.occ3d_proj import OccupancyRay


def insert_pipeline_item(cfg: DictConfig, search_type, item=None) -> None:
    if item is None:
        return
    assert OmegaConf.is_list(cfg)
    ori_cfg: List = OmegaConf.to_container(cfg)
    for index, _it in enumerate(cfg):
        if _it['type'] == search_type:
            break
    else:
        raise RuntimeError(f"cannot find type: {search_type}")
    ori_cfg.insert(index + 1, item)
    cfg.clear()
    cfg.merge_with(ori_cfg)


def draw_box_on_imgs(cfg, idx, val_input, ori_imgs, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs
    out_imgs = show_box_on_views(
        OmegaConf.to_container(cfg.dataset.object_classes, resolve=True),
        in_imgs,
        val_input['meta_data']['gt_bboxes_3d'][idx].data,
        val_input['meta_data']['gt_labels_3d'][idx].data.numpy(),
        val_input['meta_data']['lidar2image'][idx].data.numpy(),
        val_input['meta_data']['img_aug_matrix'][idx].data.numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs


def update_progress_bar_config(pipe, **kwargs):
    if hasattr(pipe, "_progress_bar_config"):
        config = pipe._progress_bar_config
        config.update(kwargs)
    else:
        config = kwargs
    pipe.set_progress_bar_config(**config)


def setup_logger_seed(cfg):
    #### setup logger ####
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    # handle log from some packages
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)
    set_seed(cfg.seed)


def build_pipe(cfg, device):
    weight_dtype = torch.float16
    if cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint = cfg.resume_from_checkpoint[:-1]
    pipe_param = {}

    model_cls = load_module(cfg.model.model_module)
    controlnet = []
    c_net_dirs = [cfg.model.controlnet_dir] if not isinstance(cfg.model.controlnet_dir,ListConfig) else cfg.model.controlnet_dir
    for i in range(len(c_net_dirs)):
        c_net_dir = c_net_dirs[i];print(c_net_dir)
        controlnet_path = os.path.join(
            cfg.resume_from_checkpoint, c_net_dir)
        logging.info(f"Loading controlnet from {controlnet_path} with {model_cls}")
        c_net = model_cls.from_pretrained(
            controlnet_path, torch_dtype=weight_dtype,low_cpu_mem_usage=False,device_map=None,
            ignore_mismatched_sizes=True)  # loading bbox_embedder for dual_branch_fg needs manual fix
        '''manually fix loading branch using map_vec_40pts'''
        if (not isinstance(cfg.use_map_vec,ListConfig) and cfg.use_map_vec and cfg.use_map_vec_40pts) or \
            (isinstance(cfg.use_map_vec,ListConfig) and cfg.use_map_vec[i] and cfg.use_map_vec_40pts[i]):
            c_net.bbox_embedder.reinitialize()
            ckpt_tmp = torch.load(controlnet_path+'/diffusion_pytorch_model.bin',map_location='cpu')
            c_net.bbox_embedder.load_state_dict(get_submodel_ckpt(ckpt_tmp,'bbox_embedder'))
            c_net.bbox_embedder.to(dtype=c_net.dtype)
            del ckpt_tmp
        '''since we don't use from_unet here, we should manually set some modules'''
        c_net.use_cam_in_temb = cfg.model.controlnet.use_cam_in_temb
        c_net.use_box_adapter = cfg.use_box_adapter
        if not c_net.use_cam_in_temb:
            c_net.adm_proj = None
        c_net.use_txt_con_fusion = cfg.model.controlnet.use_txt_con_fusion
        if not c_net.use_txt_con_fusion:
            c_net.txt_con_fusion=None
        c_net.use_txt_con_fusionp = cfg.model.controlnet.use_txt_con_fusionp
        if not c_net.use_txt_con_fusionp:
            c_net.txt_con_fusionp=None
        use_occ_3d = cfg.use_occ_3d[i] if cfg.use_dual_controlnet else cfg.use_occ_3d
        c_net.use_occ_3d = use_occ_3d
        if c_net.use_occ_3d:
            c_net.controlnet_cond_embedding = None
        c_net.eval()  # from_pretrained will set to eval mode by default
        controlnet.append(c_net)
    controlnet = controlnet[0] if not cfg.use_dual_controlnet else controlnet
    pipe_param["controlnet"] = controlnet

    if hasattr(cfg.model, "unet_module"):
        unet_cls = load_module(cfg.model.unet_module)
        unet_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.unet_dir)
        logging.info(f"Loading unet from {unet_path} with {unet_cls}")
        unet = unet_cls.from_pretrained(
            unet_path, torch_dtype=weight_dtype)
        unet.eval()
        pipe_param["unet"] = unet

    pipe_cls = load_module(cfg.model.pipe_module)
    logging.info(f"Build pipeline with {pipe_cls}")
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,  # since v1.5 has default, we need to override
        torch_dtype=weight_dtype
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    if cfg.runner.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    pipe = pipe.to(device)

    # when inference, memory is not the issue. we do not need this.
    # pipe.enable_model_cpu_offload()
    return pipe, weight_dtype


def prepare_all(cfg, device='cuda', need_loader=True):
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    setup_logger_seed(cfg)

    #### model ####
    pipe, weight_dtype = build_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    if not need_loader:
        return pipe, weight_dtype

    #### datasets ####

    if cfg.runner.validation_index == "demo":
        val_dataset = FolderSetWrapper("demo/data")
    else:
        val_dataset = build_dataset(
            OmegaConf.to_container(cfg.dataset.data.val if not cfg.gen_train_set else cfg.dataset.data.train, resolve=True)
        )
        if cfg.runner.validation_index != "all":
            val_dataset = ListSetWrapper(
                val_dataset, cfg.runner.validation_index)

    #### dataloader ####
    # our custom datasets
    num_branch = 2 if cfg.use_dual_controlnet else 1
    occ_proj = []
    map_vec = []
    for i in range(num_branch):
        use_map_vec = cfg.use_map_vec[i] if cfg.use_dual_controlnet else cfg.use_map_vec
        use_map_vec_8pts = cfg.use_map_vec_8pts[i] if cfg.use_dual_controlnet else cfg.use_map_vec_8pts
        use_map_vec_40pts = cfg.use_map_vec_40pts[i] if cfg.use_dual_controlnet else cfg.use_map_vec_40pts
        use_occ_3d = cfg.use_occ_3d[i] if cfg.use_dual_controlnet else cfg.use_occ_3d
        _map_vec = None
        if use_map_vec:
            if use_map_vec_8pts:
                _map_vec = VecMapFolderSetWrapper(train='nuscenes_map_anns_train.json',val='nuscenes_map_anns_val.json')
            elif use_map_vec_40pts:
                _map_vec = VecMapFolderSetWrapper(train='nuscenes_map_anns_train_40pts.json',val='nuscenes_map_anns_val_40pts.json')
        _occ_proj = None
        if use_occ_3d:
            compress_ratio = cfg.dataset.image_size[1] / 8 / 1600
            image_shape = cfg.dataset.back_resize
            _occ_proj = OccupancyRay(image_shape=image_shape,sample_point=320,compress_ratio=compress_ratio)
        else:
            if cfg.dataset.image_size in [[256,704],[432,768]]:
                _occ_proj = OccFolderSetWrapper('./occ_proj/occ_bg_hd',mode='image') 
            else:
                _occ_proj = OccFolderSetWrapper('./occ_proj/occ_bg',mode='image')
        occ_proj.append(_occ_proj)
        map_vec.append(_map_vec)
    occ_proj = occ_proj[0] if not cfg.use_dual_controlnet else occ_proj
    map_vec = map_vec[0] if not cfg.use_dual_controlnet else map_vec
    
    collate_fn_param = {
        "tokenizer": pipe.tokenizer,
        "template": cfg.dataset.template,
        "bbox_mode": cfg.model.bbox_mode,
        "bbox_view_shared": cfg.model.bbox_view_shared,
        "bbox_drop_ratio": cfg.runner.bbox_drop_ratio,
        "bbox_add_ratio": cfg.runner.bbox_add_ratio,
        "bbox_add_num": cfg.runner.bbox_add_num,
        "occ_proj": occ_proj, 
        "map_vec": map_vec,
        "cfg": cfg,
    }
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=partial(collate_fn, is_train=False, **collate_fn_param),
        batch_size=cfg.runner.validation_batch_size,
        num_workers=cfg.runner.num_workers,
    )
    return pipe, val_dataloader, weight_dtype


def new_local_seed(global_generator):
    local_seed = torch.randint(
        0x7ffffffffffffff0, [1], generator=global_generator).item()
    logging.debug(f"Using seed: {local_seed}")
    return local_seed


def run_one_batch_pipe(
    cfg,
    pipe: StableDiffusionBEVControlNetPipeline,
    pixel_values: torch.FloatTensor,  # useless
    captions: Union[str, List[str]],
    bev_map_with_aux: torch.FloatTensor,
    camera_param: Union[torch.Tensor, None],
    bev_controlnet_kwargs: dict,
    global_generator=None
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    # let different prompts have the same random seed
    if cfg.seed is None:
        generator = None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(torch.manual_seed(local_seed))
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [torch.manual_seed(cfg.seed)
                             for _ in range(batch_size)]
            else:
                generator = torch.manual_seed(cfg.seed)

    gen_imgs_list = [[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        image: BEVStableDiffusionPipelineOutput = pipe(
            prompt=captions,
            image=bev_map_with_aux,
            camera_param=camera_param,
            height=cfg.dataset.image_size[0],
            width=cfg.dataset.image_size[1],
            generator=generator,
            bev_controlnet_kwargs=bev_controlnet_kwargs,
            **cfg.runner.pipeline_param,
        )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return gen_imgs_list


def run_one_batch(cfg, pipe, val_input, weight_dtype, global_generator=None,
                  run_one_batch_pipe_func=run_one_batch_pipe,
                  transparent_bg=False, map_size=400):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input['meta_data']['metas'])

    # TODO: not sure what to do with filenames
    # image_names = [val_input['meta_data']['metas'][i].data['filename']
    #                for i in range(bs)]
    logging.debug(f"Caption: {val_input['captions']}")

    # map
    map_imgs = []
    for bev_map in val_input["bev_map_with_aux"]:
        map_img_np = visualize_map(cfg, bev_map, target_size=map_size)
        map_imgs.append(Image.fromarray(map_img_np))

    # ori
    ori_imgs = [None for bi in range(bs)]
    ori_imgs_with_box = [None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        ori_imgs = [
            [to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i]))
             for i in range(6)] for bi in range(bs)
        ]
        if cfg.show_box:
            ori_imgs_with_box = [
                draw_box_on_imgs(cfg, bi, val_input, ori_imgs[bi],
                                 transparent_bg=transparent_bg)
                for bi in range(bs)
            ]

    # camera_emb = self._embed_camera(val_input["camera_param"])
    camera_param = val_input["camera_param"].to(weight_dtype)

    # 3-dim list: B, Times, views
    gen_imgs_list = run_one_batch_pipe_func(
        cfg, pipe, val_input['pixel_values'], val_input['captions'],
        val_input["occ_proj"] if (cfg.task_id != '224x400') else val_input["bev_map_with_aux"], camera_param, val_input['kwargs'],
        global_generator=global_generator)

    # save gen with box
    gen_imgs_wb_list = []
    if cfg.show_box:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append([
                draw_box_on_imgs(cfg, bi, val_input, images[ti],
                                 transparent_bg=transparent_bg)
                for ti in range(len(images))
            ])
    else:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append(None)

    return map_imgs, ori_imgs, ori_imgs_with_box, gen_imgs_list, gen_imgs_wb_list
