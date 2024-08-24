from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from segment_anything import build_sam, SamPredictor 
import numpy as np
import matplotlib.pyplot as plt
import torch
import os, sys
from PIL import Image
from glob import glob
from tqdm import tqdm,trange

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def show_masks(mask,image):
    image = Image.fromarray(image).convert("RGBA")
    mask =Image.fromarray((mask.numpy()*255).astype(np.uint8)).convert("RGBA")
    return Image.alpha_composite(image,mask)

def visual_masks(masks):
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

config_path="/home/shenhongyu/Grounded-Segment-Anything/GroundingDINO_SwinB.cfg.py"
ckpt_path="/home/shenhongyu/Grounded-Segment-Anything/groundingdino_swinb_cogcoor.pth"
model = load_model(config_path,ckpt_path )

sam_checkpoint = '/home/shenhongyu/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

DATASET_PATH='/home/shenhongyu/data/scan6'
SAVE_FOLDER='sam_masks'
os.makedirs(os.path.join(DATASET_PATH,SAVE_FOLDER),exist_ok=True)

TEXT_PROMPT = "chair . table . sofa . bench . pillow"
BOX_TRESHOLD = 0.30
TEXT_TRESHOLD = 0.25

image_paths = glob_data(os.path.join(DATASET_PATH,"images","*_rgb.png"))
print('Load {} images.'.format(len(image_paths)))
for idx in trange(len(image_paths)):
    image_source, image = load_image(image_paths[idx])
    
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    plt.imsave(os.path.join(DATASET_PATH,SAVE_FOLDER,f'{idx:05d}_anno.png'),annotated_frame)

    sam_predictor.set_image(image_source)

    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    visual_mask = visual_masks(masks)
    plt.imsave(os.path.join(DATASET_PATH,SAVE_FOLDER,f'{idx:05d}_masks.png'),np.clip(visual_mask.numpy(),0,1))
    
    h, w = masks.shape[-2:]

    total_mask= torch.zeros((h, w)).to(DEVICE)
    masks=masks.to(torch.int)
    for i, mask in enumerate(masks):
        total_mask = total_mask + mask.squeeze()*(i + 20) #to prevent conflict 

    np.save(os.path.join(DATASET_PATH,SAVE_FOLDER,f'{idx:05d}_masks.npy'),total_mask.cpu().numpy())

print('Done!')
print(' Save {} sam masks at {}.'.format(len(image_paths),os.path.join(DATASET_PATH,SAVE_FOLDER)))
    
    

    
# visual
# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# annotated_frame = annotated_frame[...,::-1] # BGR to RGB
# plt.imsave('../assets/demo_ann.png',annotated_frame)
# total_mask = save_masks(masks)
# visual=show_masks(total_mask,image_source)
# visual.save('../assets/demo_ann_sam_.png')
# plt.imsave('../assets/demo_ann_sam_tt.png',total_mask.numpy())
# plt.imsave('../assets/demo_gt.png',image_source)
