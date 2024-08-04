
import torch
from scene import Scene, GaussianModel,MLP
import os

from gaussian_renderer import render

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

from utils.render_utils import save_img_f32, save_img_u8
from utils.render_utils import generate_path, create_videos

#import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def saveRGB(label,path,k=12):
    img=label.view(384,384,-1)
    img=img.argmax(dim=2, keepdim=True).expand(-1, -1, 3)
    bg=(img==k)
    id_vis=img*255/(k+1)
    id_vis[:,:,1]=(id_vis[:,:,1]*23)%256
    id_vis[:,:,2]=(id_vis[:,:,2]*13)%256

    id_vis=id_vis*(~bg).int()+bg.int()*255

    id_vis=np.array(id_vis.cpu(),dtype=np.uint8)
    
    plt.imsave(path,id_vis, cmap='viridis')


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
    parser.add_argument("--start_checkpoint", type=str, default = "output/scan6/chkpnt_with_feature2000.pth")
    parser.add_argument("--classifier_checkpoint", type=str, default = "output/scan6/classifier_chkpnt2000.pth")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,shuffle=False)

    k=12#TODO
    classifier=torch.load(args.classifier_checkpoint)

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
        inverse=False
        object=get_object_by_id(gaussians,obj_id,inverse)
        if inverse:
            obj_id*=-1
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

    render_id_map=True
    if render_id_map:
        obj_id=12
        inverse=False
        object=get_object_by_id(gaussians,obj_id,inverse)
        if inverse:
            obj_id*=-1
        vis_path="output/scan6/objects/mask/"
        path=os.path.join(vis_path,str(obj_id))
        os.makedirs(path,exist_ok=True)
        with torch.no_grad():
            viewpoint_stack = scene.getTrainCameras()
            for idx, viewpoint in enumerate(viewpoint_stack):
                render_pkg = render(viewpoint, object, pipe, background,True)
                instance_image=render_pkg["instance_image"]
                label=classifier(instance_image.permute(1, 2, 0)).view(-1,k+1)
                save_path=os.path.join(path,f'id{idx:05d}.png')
                saveRGB(label,save_path)





