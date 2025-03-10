# DualDiff
A dual-branch conditional diffusion model designed to enhance driving scene generation across multiple views and video sequences.

<!-- ## Demo Video
<video width="800" controls>
  <source src="https://github.com/yangzhaojason/DualDiff/raw/refs/heads/main/media/5721_1739810373.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video> -->

<!-- <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Demo Video</title>
</head>
<body>
    <h2>Demo Video</h2>
    <video width="700" controls autoplay loop muted>
        <source src="https://your-video-url.com/demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</body>
</html> -->

## Demo Video
<video width="700" controls>
  <source src="https://yangzhaojason.github.io/DualDiff/media/5721_1739810373.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Abstract
Abstract—Accurate and high-fidelity driving scene reconstruction demands the effective utilization of comprehensive scene information as conditional inputs. Existing methods predominantly rely on 3D bounding boxes and BEV road maps for
foreground and background control, which fail to capture the full complexity of driving scenes and adequately integrate multimodal information. In this work, we present DualDiff, a dual-branch conditional diffusion model designed to enhance driving scene generation across multiple views and video sequences. Specifically, we introduce Occupancy Ray-shape Sampling (ORS) as a conditional input, offering rich foreground and background semantics alongside 3D spatial geometry to precisely control the generation of both elements. To improve the synthesis of finegrained foreground objects, particularly complex and distant ones, we propose a Foreground-Aware Mask (FGM) denoising loss function. Additionally, we develop the Semantic Fusion Attention (SFA) mechanism to dynamically prioritize relevant information and suppress noise, enabling more effective multimodal fusion. Finally, to ensure high-quality image-to-video generation, we introduce the Reward-Guided Diffusion (RGD) framework, which maintains global consistency and semantic coherence in generated videos. Extensive experiments demonstrate that DualDiff achieves state-of-the-art (SOTA) performance across multiple datasets. On the NuScenes dataset, DualDiff reduces the FID score by 4.09% compared to the best baseline. In downstream tasks, such as BEV segmentation, our method improves vehicle mIoU by 4.50% and road mIoU by 1.70%, while in BEV 3D object detection, the foreground mAP increases by 1.46%.
## Installation


## Citation
If you find our work useful, please cite our paper:

```bibtex
@article{yang2025dualdiff+,
  title={DualDiff+: Dual-Branch Diffusion for High-Fidelity Video Generation with Reward Guidance},
  author={Yang, Zhao and Qian, Zezhong and Li, Xiaofan and Xu, Weixiang and Zhao, Gongpeng and Yu, Ruohong and Zhu, Lingsi and Liu, Longjun},
  journal={arXiv preprint arXiv:2503.03689},
  year={2025}
}