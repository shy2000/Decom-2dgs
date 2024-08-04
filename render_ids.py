
import torch
from scene import Scene
import os

from gaussian_renderer import render

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.render_utils import save_img_f32, save_img_u8
from utils.render_utils import generate_path, create_videos

import open3d as o3d




def get_object_by_id(gaussian,idx,inverse=False):
    ids=gaussian.get_ids
    unique_values, inverse_indices = ids.unique(return_inverse=True)
    value_indices = {value.item(): (inverse_indices == i).nonzero(as_tuple=True)[0] for i, value in enumerate(unique_values)}
    if idx > len(value_indices):
        raise ValueError("idx error!")
    #for i in range(13):
    #    print(len(value_indices[i])) 
    if inverse:
        mask = torch.ones(len(ids), dtype=torch.bool)
        mask[value_indices[idx]] = False  
    else:
        mask=value_indices[idx]
    gaus=gaussian.capture('Objects',mask)
    obj=GaussianModel(gaussian.max_sh_degree)
    obj.restore(gaus,mode='obj')
    return obj
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,shuffle=False)
    checkpoint=args.start_checkpoint
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        scene.loaded_iter=first_iter
        gaussians.restore(model_params,mode='render')
        #print(gaussians.get_opacity.shape[0])
    else:
       raise ValueError("checkpoint missing!")
    #chpt
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))

    render_objects=False
    if render_objects:
        obj_id=5
        object=get_object_by_id(gaussians,obj_id,inverse=False)
        vis_path="output/scan6/objects/"
        path=os.path.join(vis_path,str(obj_id))
        os.makedirs(path,exist_ok=True)
        with torch.no_grad():
            viewpoint_stack = scene.getTrainCameras()
            for idx, viewpoint in enumerate(viewpoint_stack):
                render_pkg = render(viewpoint, object, pipe, background,False)
                img=render_pkg["render"]
                #print(img.shape)
                save_img_u8(img.permute(1,2,0).cpu().detach().numpy(),os.path.join(path,str(idx)+".png"))

    get_bg=True
    if get_bg:




