from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamPredictor ,SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import torch
import os, sys
from PIL import Image
from glob import glob
from tqdm import tqdm,trange
import cv2

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def show_masks(mask,image):
    image = Image.fromarray(image).convert("RGBA")
    mask =Image.fromarray((mask.numpy()*255).astype(np.uint8)).convert("RGBA")
    return Image.alpha_composite(image,mask)

def save_masks(masks):
    h, w = masks.shape[-2:]
    masks=masks.cpu()
    total_mask= torch.zeros((h, w, 4))
    for i in masks:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        mask_image = i.reshape(h, w, 1) * color.reshape(1, 1, -1)
        total_mask = total_mask + mask_image
    return total_mask

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img
    
    #ax.imshow(img)

sam_checkpoint = '/home/shenhongyu/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
mask_generator=SamAutomaticMaskGenerator(
    model=sam,        
    points_per_side=4,
    pred_iou_thresh=0.7,
    box_nms_thresh=0.7,
    stability_score_thresh=0.85,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
    )
mask_generator2=SamAutomaticMaskGenerator(
    model=sam,        
    points_per_side=8,
    pred_iou_thresh=0.7,
    box_nms_thresh=0.7,
    stability_score_thresh=0.85,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
    )
mask_generator3=SamAutomaticMaskGenerator(
    model=sam,        
    points_per_side=16,
    pred_iou_thresh=0.7,
    box_nms_thresh=0.7,
    stability_score_thresh=0.85,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
    )

path=r'/home/shenhongyu/data/scan6/images/000036_rgb.png'
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
plt.imsave("../assets/only_sam_mask.png",show_anns(masks))
masks2 = mask_generator2.generate(image)
plt.imsave("../assets/only_sam_mask2.png",show_anns(masks2))
masks3 = mask_generator3.generate(image)
plt.imsave("../assets/only_sam_mask3.png",show_anns(masks3))


# plt.figure(figsize=(20,20))
# plt.imshow(image)
#show_anns(masks)
# plt.axis('off')
# plt.show() 
