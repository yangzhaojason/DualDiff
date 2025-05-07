import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer
import cv2
from typing import Tuple

import warnings
warnings.filterwarnings("ignore")

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def view_images(images, view, savepath, num_rows=1, offset_ratio=0.02):
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

    pil_img.save(os.path.join(savepath,f'view_images_{view}.png'))
    # display(pil_img)

def process(token,ip=''):
    # map_ = torch.load(r'/opt/data/private/aigc/MagicDrive_official/A driving scene image at singapore-onenorth. Barriers, peds loading car trunk, luggage on sidewalk, parked bicycles, lane change.pth')
    sep = '.' if len(ip) == 0 else ip
    map_ = torch.load(f'../explore/{token}/{sep}/{token}_attn.pth')

    t_list = list(map_['unet'].keys())
    layer_list_c = list(map_['cnet'][0][t_list[0]].keys()) # map_['cnet']位置0为常规attn，位置1为adapter attn，此处为空
    layer_list_u = list(map_['unet'][t_list[0]].keys())
    
    # collect_cnet = {}
    # for l in layer_list_c:
    #     collect_cnet[l] = []
    # for t in t_list:
    #     for l in layer_list_c:
    #         collect_cnet[l].append(torch.cat([map_['cnet'][0][t][l],map_['cnet'][1][t][l]],axis=2))
    
    collect_cnet = {}
    for l in layer_list_c:
        # if 'up' in l:
        collect_cnet[l] = []
    for t in t_list:
        for l in layer_list_c:
            # if 'up' in l:
            collect_cnet[l].append(map_['cnet'][0][t][l])

    collect_unet = {}
    for l in layer_list_u:
        if 'up' in l:
            collect_unet[l] = []
    for t in t_list:
        for l in layer_list_u:
            if 'up' in l:
                collect_unet[l].append(map_['unet'][t][l])

    # 96: channels
    # 1400: 50*28 feature reshape
    # 106: 1cam + 77prompt + 28box

    # if visualizing cnet
    # collect_unet = collect_cnet

    res = (28,50)
    gather_layer = []
    for i in list(collect_unet.keys()):
        # print(i, len(collect_unet[i]), collect_unet[i][0].shape)  # layername, timesteps, torch.Size([6*attn channels, res[0]*res[1], 106])
        tmp = [_.reshape(6,-1,*_.shape[1:]).mean(1) for _ in collect_unet[i]]  # avg over attn channels, list of timesteps * torch.Size([6, res[0]*res[1], 106])
        tmp = torch.stack(tmp).mean(0)  # avg over timesteps, torch.Size([6, res[0]*res[1], 106])
        if tmp.shape[1] == res[0]*res[1] and not i.startswith('mid'):
            gather_layer.append(tmp)
    gather_layer = torch.stack(gather_layer).mean(0)  # avg over layers
    print(gather_layer.shape)
    with open(f'../explore/{token}/{sep}/{token}_attn.txt', 'r') as f:
        prompt = f.read()
    print(prompt)
    print(token)
    print('='*len(token))
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode  # compared to input prompts, words are further split into seperate tokens (fine grain)
    for view in range(6):
        images = []
        gather_layer_view = gather_layer[view]
        for i in range(1,len(tokens)+1):  # there is a cam in the front, followed by 77 tokens, followed by 28 box
            image = gather_layer_view[:, i]
            image = image.reshape(res[0],res[1],-1)
            image = image.numpy().astype(np.float)
            # image = image/image.max()
            image = (image - image.min()) / (image.max() - image.min())
            cmap = plt.get_cmap('coolwarm')  # 选择颜色映射
            colored_image = np.zeros((res[0], res[1], 4), dtype=np.float)
            for j in range(res[0]):
                for k in range(res[1]):
                    colored_image[j, k] = cmap(image[j, k])
            mapped_data_rgb = (colored_image[:, :, :3] * 255).astype(np.uint8)
            image = np.array(Image.fromarray(mapped_data_rgb).resize((400, 224)))#(256, 256, 3)
            image = text_under_image(image, decoder(int(tokens[i-1])))
            images.append(image)
        view_images(np.stack(images, axis=0),view=view,savepath=os.path.join('../explore',token,ip),num_rows=4)

    for view in range(6):
        images = []
        gather_layer_view = gather_layer[view]
        for i in range(1+77,gather_layer_view.shape[-1]):  # there is a cam in the front, followed by 77 tokens, followed by <28 box
            image = gather_layer_view[:, i]
            image = image.reshape(res[0],res[1],-1)
            image = image.numpy().astype(np.float)
            # image = image/image.max()
            image = (image - image.min()) / (image.max() - image.min())
            cmap = plt.get_cmap('coolwarm')  # 选择颜色映射
            colored_image = np.zeros((res[0], res[1], 4), dtype=np.float)
            for j in range(res[0]):
                for k in range(res[1]):
                    colored_image[j, k] = cmap(image[j, k])
            mapped_data_rgb = (colored_image[:, :, :3] * 255).astype(np.uint8)
            image = np.array(Image.fromarray(mapped_data_rgb).resize((400, 224)))#(256, 256, 3)
            images.append(image)
        view_images(np.stack(images, axis=0),view=str(view)+'box',savepath=os.path.join('../explore',token,ip),num_rows=1)

if __name__ == '__main__':
    tokenizer = CLIPTokenizer.from_pretrained('../pretrained/stable-diffusion-v1-5', subfolder="tokenizer")
    for token in [
        '53bc14f0ee1f459e96250c3f631eeba4',
        'a7831d4d1db54053a501d0418545fee2',
        'a9460d1306f94aaf894af8b07d5f23a9',
        'c6d18bda94224330b898c3c2ff32fa74',
        'd7188ab6f6494e3e900f16efa4c4a7fa',
        'edd33328448f40b49f0f7e1da07b9ca8',
        ]:
        process(token,ip='crop')
