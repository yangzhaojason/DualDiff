import logging
import os
import contextlib
from omegaconf import OmegaConf, ListConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat

from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler

from ..misc.common import load_module, convert_outputs_to_fp16, move_to
from .base_runner import BaseRunner
from .utils import smart_param_count
from ..networks.map_embedder import BEVControlNetConditioningEmbedding
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ..networks.unet_addon_rawbox import get_submodel_ckpt


class ControlnetUnetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, controlnet, unet, controlnet_refer=None, weight_dtype=torch.float32,
                 unet_in_fp16=True) -> None:
        super().__init__()
        self.controlnet = controlnet
        if isinstance(self.controlnet,list):
            self.c_net1 = self.controlnet[0]  # this only 'expose' the controlnet inside the list out
            self.c_net2 = self.controlnet[1]  # and can be found & initialized by accelerator.prepare
        self.unet = unet
        # self.controlnet_refer = controlnet_refer
        self.weight_dtype = weight_dtype
        self.unet_in_fp16 = unet_in_fp16

    def forward(self, noisy_latents, timesteps, camera_param,
                encoder_hidden_states, encoder_hidden_states_uncond,
                controlnet_image, controlnet_image_occ, **kwargs):
        N_cam = noisy_latents.shape[1]
        kwargs = move_to(
            kwargs, self.weight_dtype, lambda x: x.dtype == torch.float32)
        bboxes_3d_data=kwargs['bboxes_3d_data']
        del kwargs['bboxes_3d_data']

        if isinstance(self.controlnet, list):
            for i in range(len(self.controlnet)):
                down_samples, mid_sample, \
                states_with_cam = self.controlnet[i](
                    noisy_latents,  # b, N_cam, 4, H/8, W/8
                    timesteps,  # b
                    camera_param=camera_param,  # b, N_cam, 189
                    encoder_hidden_states=encoder_hidden_states,  # b, len, 768
                    encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
                    controlnet_cond=controlnet_image_occ[i],  # b, 3, 224, 2400(occ proj)
                    return_dict=False,
                    bboxes_3d_data=bboxes_3d_data[i],
                    **kwargs,
                )
                # merge samples
                if i == 0:
                    down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
                else:
                    down_block_res_samples = [
                        samples_prev + samples_curr
                        for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                    ]
                    mid_block_res_sample += mid_sample
                if i == 0:
                    encoder_hidden_states_with_cam = states_with_cam # take this from 1st controlnet
        else:
            if isinstance(self.controlnet.controlnet_cond_embedding, BEVControlNetConditioningEmbedding):
                controlnet_cond = controlnet_image  # vanilla
            else:
                controlnet_cond = controlnet_image_occ
            # fmt: off
            down_block_res_samples, mid_block_res_sample, \
            encoder_hidden_states_with_cam = self.controlnet(
                noisy_latents,  # b, N_cam, 4, H/8, W/8
                timesteps,  # b
                camera_param=camera_param,  # b, N_cam, 189
                encoder_hidden_states=encoder_hidden_states,  # b, len, 768
                encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
                controlnet_cond=controlnet_cond,  # b, 3, 224, 2400(occ proj)
                return_dict=False,
                bboxes_3d_data=bboxes_3d_data,
                **kwargs,
            )
            # fmt: on

        # starting from here, we use (B n) as batch_size
        noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
        if timesteps.ndim == 1:
            timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            model_pred = self.unet(
                noisy_latents,  # b x n, 4, H/8, W/8
                timesteps.reshape(-1),  # b x n
                encoder_hidden_states=encoder_hidden_states_with_cam.to(
                    dtype=self.weight_dtype
                ),  # b x n, len + 1, 768
                # TODO: during training, some camera param are masked.
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype)
                    for sample in down_block_res_samples
                ],  # all intermedite have four dims: b x n, c, h, w
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.weight_dtype
                ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
            ).sample

        model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
        return model_pred#, down_block_res_samples_refer+[mid_block_res_sample_refer], down_block_res_samples+[mid_block_res_sample]


class MultiviewRunner(BaseRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)

    def _init_fixed_models(self, cfg):
        # fmt: off
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        # fmt: on

    def _init_trainable_models(self, cfg):
        # fmt: off
        unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="unet")
        # fmt: on

        model_cls = load_module(cfg.model.unet_module)
        # unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        # self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)
        self.unet =  model_cls.from_pretrained(os.path.join(cfg.official_ckpt,'unet'),torch_dtype=torch.float16)

        model_cls = load_module(cfg.model.model_module)
        controlnet_param = OmegaConf.to_container(
            self.cfg.model.controlnet, resolve=True)
        # self.controlnet = model_cls.from_unet(unet, **controlnet_param)
        self.controlnet_refer = model_cls.from_pretrained(os.path.join(cfg.official_ckpt,'controlnet'),
                                                          torch_dtype=torch.float32, low_cpu_mem_usage=False, device_map=None) # ought to be no mismatched modules
        # ckpt_seg = torch.load(cfg.controlnet_seg_pretrained,map_location='cpu')
        # self.controlnet = model_cls.from_unet(unet, seg_init=ckpt_seg, **controlnet_param)
        num_branch = 2 if cfg.use_dual_controlnet else 1; self.controlnet = []
        for i in range(num_branch):
            if cfg.task_id == '224x400': # vanilla
                c_net = self.controlnet_refer
            else:
                if not cfg.use_trained_weights:
                    c_net_path = cfg.controlnet_seg_pretrained
                else:
                    c_net_paths = cfg.trained_weights
                    c_net_path = c_net_paths[i] if isinstance(c_net_paths, ListConfig) else c_net_paths
                c_net = model_cls.from_pretrained(c_net_path,torch_dtype=torch.float32,
                    low_cpu_mem_usage=False, ignore_mismatched_sizes=True) # skip initialization of the mismatched modules
            '''manually fix loading branch using map_vec_40pts'''
            if (not isinstance(cfg.use_map_vec,ListConfig) and cfg.use_map_vec and cfg.use_map_vec_40pts) or \
                (isinstance(cfg.use_map_vec,ListConfig) and cfg.use_map_vec[i] and cfg.use_map_vec_40pts[i]):
                c_net.bbox_embedder.reinitialize()
                if cfg.use_trained_weights:
                    ckpt_tmp = torch.load(c_net_path+'/diffusion_pytorch_model.bin',map_location='cpu')
                    c_net.bbox_embedder.load_state_dict(get_submodel_ckpt(ckpt_tmp,'bbox_embedder'))
                    c_net.bbox_embedder.to(dtype=c_net.dtype)
                    del ckpt_tmp
            '''since we don't use from_unet here, we should manually set some modules'''
            c_net.use_cam_in_temb = controlnet_param['use_cam_in_temb']
            c_net.use_box_adapter = cfg.use_box_adapter
            if not c_net.use_cam_in_temb:
                c_net.adm_proj = None
            c_net.use_txt_con_fusion = controlnet_param['use_txt_con_fusion']
            if not c_net.use_txt_con_fusion:
                c_net.txt_con_fusion = None
            c_net.use_txt_con_fusionp = controlnet_param['use_txt_con_fusionp']
            if not c_net.use_txt_con_fusionp:
                c_net.txt_con_fusionp = None
            use_occ_3d = cfg.use_occ_3d[i] if cfg.use_dual_controlnet else cfg.use_occ_3d
            c_net.use_occ_3d = use_occ_3d
            if c_net.use_occ_3d:
                c_net.controlnet_cond_embedding = None
            
            if not cfg.use_trained_weights:  
                c_net.bbox_embedder.load_state_dict(self.controlnet_refer.bbox_embedder.state_dict())
                c_net.cam2token.load_state_dict(self.controlnet_refer.cam2token.state_dict())
            self.controlnet.append(c_net)
        if not cfg.task_id == '224x400':
            del self.controlnet_refer
        self.controlnet = self.controlnet[0] if not cfg.use_dual_controlnet else self.controlnet

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if isinstance(self.controlnet, list):
            for c_net in self.controlnet:
                c_net.train(train)
        else:
            self.controlnet.train(train)
        self.unet.requires_grad_(False)
        for name, mod in self.unet.trainable_module.items():
            logging.debug(
                f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)

    def _set_xformer_state(self):
        # xformer
        if self.cfg.runner.enable_xformers_memory_efficient_attention or self.cfg.use_box_adapter:
            import xformers
            self.unet.enable_xformers_memory_efficient_attention()
            if not self.cfg.use_box_adapter:
                if isinstance(self.controlnet, list):
                    for c_net in self.controlnet:
                        c_net.enable_xformers_memory_efficient_attention()
                else:
                    self.controlnet.enable_xformers_memory_efficient_attention()
            else:
                assert not self.cfg.use_dual_controlnet
                from ..networks.box_adapter import box_adapter
                self.controlnet = box_adapter(self.controlnet)

    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        if isinstance(self.controlnet, list):
            params_to_optimize = []
            for c_net in self.controlnet:
                params_to_optimize += list(c_net.parameters())
        else:
            params_to_optimize = list(self.controlnet.parameters())
        unet_params = self.unet.trainable_parameters
        param_count = smart_param_count(unet_params)
        logging.info(
            f"[MultiviewRunner] add {param_count} params from unet to optimizer.")
        params_to_optimize += unet_params
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        # fmt: off
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )
        # fmt: on

    def prepare_device(self):
        self.controlnet_unet = ControlnetUnetWrapper(self.controlnet, self.unet)
        # accelerator
        ddp_modules = (
            self.controlnet_unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.controlnet_unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = ddp_modules

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        # self.controlnet_refer.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name, mod in self.unet.trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(
                        dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32.")
        controlnet_unet = self.accelerator.unwrap_model(self.controlnet_unet)
        controlnet_unet.weight_dtype = self.weight_dtype
        controlnet_unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        with torch.no_grad():
            if isinstance(self.controlnet, list):
                for c_net in self.controlnet:
                    self.accelerator.unwrap_model(c_net).prepare(
                        self.cfg,
                        tokenizer=self.tokenizer,
                        text_encoder=self.text_encoder
                    )
            else:
                self.accelerator.unwrap_model(self.controlnet).prepare(
                    self.cfg,
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder
                )

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        if isinstance(self.controlnet, list):
            for i in range(len(self.controlnet)):
                c_net = self.accelerator.unwrap_model(self.controlnet[i])
                c_net.save_pretrained(
                    os.path.join(root, self.cfg.model.controlnet_dir[i]))
        else:
            controlnet = self.accelerator.unwrap_model(self.controlnet)
            controlnet.save_pretrained(
                os.path.join(root, self.cfg.model.controlnet_dir))
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        logging.info(f"Save your model to: {root}")

    def _train_one_stop(self, batch):
        self.controlnet_unet.train()
        # self.controlnet_refer.train(False)
        # self.controlnet.bbox_embedder.train(False)
        # self.controlnet.cam2token.train(False)
        with self.accelerator.accumulate(self.controlnet_unet):
            N_cam = batch["pixel_values"].shape[1]

            # Convert images to latent space
            latents = self.vae.encode(
                rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)
            # camera_emb = self._embed_camera(batch["camera_param"])
            camera_param = batch["camera_param"].to(self.weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            if self.cfg.model.train_with_same_t:
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
            else:
                timesteps = torch.stack([torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ) for _ in range(N_cam)], dim=1)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(
                batch
                ["uncond_ids"])[0]

            controlnet_image = batch["bev_map_with_aux"].to(
                dtype=self.weight_dtype)
            if isinstance(self.controlnet_unet.module.controlnet, list):
                controlnet_image_occ = []
                for i in range(len(self.controlnet_unet.module.controlnet)):
                    controlnet_image_occ.append(batch["occ_proj"][i].to(
                        dtype=self.weight_dtype))
            else:
                controlnet_image_occ = batch["occ_proj"].to(
                    dtype=self.weight_dtype)

            model_pred = self.controlnet_unet(
                noisy_latents, timesteps, camera_param, encoder_hidden_states,
                encoder_hidden_states_uncond, controlnet_image, controlnet_image_occ,
                **batch['kwargs'],
            )

            # directly predict original image from latents
            if self.cfg.use_tone_guidance: # activate according to timestep?
                res={'pred':[],'gt':[]}
                tone_guidance = 0
                for _bs in range(model_pred.shape[0]):
                    pred_original_sample = self.noise_scheduler.step(model_pred[[_bs]],timesteps[_bs],noisy_latents[[_bs]]).pred_original_sample
                    pred_image = self.decode_latents(pred_original_sample)
                    # pred_images.append(pred_image)
                    pred_mu = self.mscn(pred_image)
                    gt_mu = self.mscn(batch["pixel_values"][[_bs]])
                    tone_guidance += F.mse_loss(pred_mu.float(), gt_mu.float(), reduction='mean')
                    # check images
                    # pred_image = pred_images[-1]
                    # gt_image = batch["pixel_values"][_bs]
                    res['pred'].append(pred_mu.detach().cpu())
                    res['gt'].append(gt_mu.detach().cpu())
                    # torch.save(torch.stack([pred_image[0].detach().cpu(),gt_image.cpu()]),'first_stage.pth')
                torch.save(res,'first_stage.pth')
                # pred_images = torch.cat(pred_images,dim=0)
                # pred_mu = self.mscn(pred_images)
                # gt_mu = self.mscn(batch["pixel_values"])
                # tone_guidance = F.mse_loss(pred_mu.float(), gt_mu.float(), reduction='mean')
                tone_guidance /= model_pred.shape[0]

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            # # knowledge_distillation
            # # down_block_res_samples(n_cam,channels,res0,res1)
            # # [6, 320,28,50]*3
            # # [6, 320/640,14,25]*3
            # # [6,640/1280, 7,13]*3
            # # [6,1280, 4, 7]*3
            # # mid_block_res_sample
            # # [6,1280, 4, 7]
            # heatmap_mask = [rearrange(i,'b n ... -> (b n) ...') for i in batch['heatmap_gt']] # bs,n_cam,max_n_box,28,50
            # # self.visualize_heatmap(batch,gt=True)
            # kd_loss = 0
            # for i_level in range(len(res_gt)):
            #     loss_level = F.mse_loss(res_pred[i_level].float(), res_gt[i_level].float(), reduction='none')
            #     loss_level = loss_level * heatmap_mask[i_level//3].unsqueeze(1) # broadcast
            #     kd_loss += loss_level.mean()
            # kd_loss = kd_loss / len(res_gt)

            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            if not self.cfg.use_aug_loss:
                loss = loss.mean() #+ aug_loss.mean()
            else:
                aug_loss = loss * batch['heatmap_gt'][0].unsqueeze(2)
                loss = loss.mean() + aug_loss.mean()
            if self.cfg.use_tone_guidance:
                loss = loss + 2 * tone_guidance
            # loss = loss + 0.01 * kd_loss

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = self.controlnet_unet.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)

        return loss
    
    def visualize_heatmap(self,ret_dict,gt=False):
        to_visualize = [i.detach().cpu() for i in ret_dict['heatmap_gt']]
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
                    image = np.array(Image.fromarray(mapped_data_rgb).resize((400, 224)))#(256, 256, 3)
                    images.append(image)
            self.view_images(np.stack(images, axis=0),view=f'{tokens}_{cams}',savepath=f'./heatmap_{gt}_viz', save=True, num_rows=len(to_visualize))
    
    def view_images(self, images, view, savepath, num_rows=1, offset_ratio=0.02, save=True):
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

    def decode_latents(self, latents):
        # decode latents with 5-dims
        latents = 1 / self.vae.config.scaling_factor * latents

        bs = len(latents)
        latents = rearrange(latents, 'b c ... -> (b c) ...')
        image = self.vae.decode(latents.to(dtype=self.vae.dtype)).sample
        image = rearrange(image, '(b c) ... -> b c ...', b=bs)

        image = (image / 2 + 0.5).clamp(0, 1) # 1,6,3,224,400
        # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = rearrange(image.cpu(), '... c h w -> ... h w c').float().numpy()
        return image

    def mscn(self,rgb):
        # rgb2yuv, input as b,n_cam,c,h,w
        rgb = rgb * 0.5 + 0.5
        A = torch.tensor([[0.299, -0.14714119,0.61497538], 
                        [0.587, -0.28886916, -0.51496512],
                        [0.114, 0.43601035, -0.10001026]])   # from  Wikipedia
        A = A.to(rgb.device, dtype=rgb.dtype)
        yuv = torch.tensordot(rgb,A,dims=([2],[1])) # b,n,[c],h,w <tensorbdot> 3,[3] -> b,n,h,w,3
        y = yuv[...,0].unsqueeze(2) # b,n,1,h,w
        y = rearrange(y, 'b n ... -> (b n) ...') # b*n,1,h,w
        mu = torchvision.transforms.GaussianBlur(kernel_size=17,sigma=17/6)(y)
        return mu