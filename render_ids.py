
import torch
import torch.nn as nn
from scene import Scene, GaussianModel,MLP
import os
import joblib

from gaussian_renderer import render

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

from utils.render_utils import save_img_f32, save_img_u8
from utils.render_utils import generate_path, create_videos
from tqdm import trange

#import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import hdbscan

#{"chair_1": 32, "table_1": 38, "chair_2": 41, "table_2": 43, "sofa_1": 44, "sofa_2": 51, "sofa_3": 58, "sofa_4": 65, "table_3": 75, "chair_3": 80, "chair_4": 90, "chair_5": 93}

def obj_cluster(features,output_save_path,cluster_save_path,recluster):
    if os.path.exists(output_save_path) and not recluster:
        labels=np.load(output_save_path)
        return labels
    if os.path.exists(cluster_save_path) and not recluster:
        clusterer = joblib.load(cluster_save_path)
    else:
        clusterer = hdbscan.HDBSCAN(metric='euclidean',min_cluster_size=3000,core_dist_n_jobs=-1).fit(features)
        joblib.dump(clusterer, cluster_save_path)
 
    labels=clusterer.labels_
    np.save(output_save_path,labels)
    return labels

def saveRGB(label,path,class_n=12):
    img=label.view(384,384,-1)
    if class_n>0:
        img=img.argmax(dim=2, keepdim=True).expand(-1, -1, 3)
        bg=(img==class_n)
    else:
        img=img.expand(-1,-1,3)
        bg=(img<0)

    id_vis=img*255/(class_n+1)
    id_vis[:,:,1]=(id_vis[:,:,1]*23)%256
    id_vis[:,:,2]=(id_vis[:,:,2]*13)%256

    id_vis=id_vis*(~bg).int()+bg.int()*255

    id_vis=np.array(id_vis.cpu(),dtype=np.uint8)
    
    plt.imsave(path,id_vis, cmap='viridis')

def get_obj_by_mask(gaussian,mask,inverse=False):
    if inverse:
        mask=~mask
    gaus=gaussian.capture('Objects',mask)
    obj=GaussianModel(gaussian.max_sh_degree)
    obj.restore(gaus,mode='obj')
    #print("get {} gaussians".format(len(obj._opacity)))
    return obj

def get_object_by_id(gaussian,idx,inverse=False):
    ids=gaussian.get_ids
    mask=(ids==idx)
    return get_obj_by_mask(gaussian,mask,inverse)

def get_object_by_feature(gaussian,querys,inverse=False,threshold_sim=1e-5,threshold_dist=0.7):
    features=gaussian.get_instance_feature
    pdist = nn.PairwiseDistance(p=2)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6) 
    final_mask = torch.zeros(len(features), dtype=torch.bool,device='cuda')
    for query in querys:
        sims=cos(features,query)  
        mask1 = (sims<threshold_sim)
        dist=pdist(features[mask1],query)
        mask2=(dist<threshold_dist)
        #print(mask2.sum())
        final_mask[mask1] = final_mask[mask1] | mask2
        # if int(mask2.sum()/1000)==60:
        #     final_mask[mask1] = mask2
        #     np.save(os.path.join(f'table.npy'),query.cpu().numpy())
        #     break

    return get_obj_by_mask(gaussian,final_mask,inverse)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--start_checkpoint", type=str, default = "output/chkpnt/replica/scan6/chkpnt_contrastive_2000.pth")
    parser.add_argument("--classifier_checkpoint", type=str, default = "output/scan6/classifier_chkpnt2000.pth")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,shuffle=False)

    class_n=12#TODO
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
    viewpoints = scene.getTrainCameras()
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

    render_id_map=False
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
                label=classifier(instance_image.permute(1, 2, 0)).view(-1,class_n+1)
                save_path=os.path.join(path,f'id{idx:05d}.png')
                saveRGB(label,save_path)

    render_objects_feature=False
    if render_objects_feature:
        obj_id=4
        n_sample=100
        
        inverse=False
        vis_path="output/scan6/objects_f/"
        path=os.path.join(vis_path,str(obj_id))
        os.makedirs(path,exist_ok=True)
        with torch.no_grad():
            view = viewpoints[99]
            render_pkg = render(view, gaussians, pipe, background,True)
            instance_image=render_pkg["instance_image"]
            instance_image=instance_image.permute(1, 2, 0)
            img=render_pkg["render"]
            save_img_u8(img.permute(1,2,0).cpu().detach().numpy(),os.path.join(path,"view.jpg"))
            label=classifier(instance_image)
            id_img=label.argmax(dim=2, keepdim=True)


            mask=(id_img == obj_id).squeeze(-1)
            obj_features=instance_image[mask].view(-1,16)

            if n_sample> len(obj_features):
                 raise ValueError('n_sample should smaller than {}'.format(len(obj_features)))
            indices = torch.randint(0, obj_features.size(0), (n_sample,)).cuda()
            query=torch.index_select(obj_features, 0, indices)#.mean().unsqueeze(0)

            object=get_object_by_feature(gaussians,query,inverse,threshold_sim=1e-8,threshold_dist=1)
            for idx, viewpoint in enumerate(viewpoints):
                render_pkg = render(viewpoint, object, pipe, background,True)
                img=render_pkg["render"]
                save_img_u8(img.permute(1,2,0).cpu().detach().numpy(),os.path.join(path,"RGB_"+str(idx)+".png"))
                instance_image=render_pkg["instance_image"]
                label=classifier(instance_image.permute(1, 2, 0)).view(-1,class_n+1)
                save_path=os.path.join(path,f'id{idx:05d}.png')
                saveRGB(label,save_path)
    
    cluster=False
    with torch.no_grad():#with classifier
        if cluster:
            output_save_path='cluster_labels_1.npy'
            cluster_save_path='hdbscan_model_1.pkl'
            labels=obj_cluster(gaussians.get_instance_feature.cpu(),output_save_path,cluster_save_path)

            vis_path="output/scan6/objects_s2/"
            for obj_id in trange(np.max(labels)):
                path=os.path.join(vis_path,str(obj_id))
                mask=(labels==obj_id)
                object=get_obj_by_mask(gaussians,mask)

                for idx, viewpoint in enumerate(viewpoints):
                    render_pkg = render(viewpoint, object, pipe, background,True)
                    instance_image=render_pkg["instance_image"]
                    img=render_pkg["render"]
                    label=classifier(instance_image.permute(1, 2, 0)).view(-1,class_n+1)
                    id_img=label.argmax(dim=-1, keepdim=True)
                    bg=(id_img==class_n)
                    bg=~bg
                    if img.sum() < 200 or bg.sum() < 50:
                        continue      
                    os.makedirs(path,exist_ok=True)
                    save_img_u8(img.permute(1,2,0).cpu().detach().numpy(),os.path.join(path,"RGB_"+str(idx)+".png"))
                    save_path=os.path.join(path,f'id{idx:05d}.png')
                    saveRGB(label,save_path)
    
    contrastive=True
    with torch.no_grad():
        if contrastive:
            output_save_path='cluster_labels_2.npy'
            cluster_save_path='hdbscan_model_2.pkl'
            recluster=1
            all_labels=obj_cluster(gaussians.get_instance_feature.cpu(),output_save_path,cluster_save_path,recluster)
            #clusterer = joblib.load(cluster_save_path)
            # clusterer.generate_prediction_data()
            # label,_ = hdbscan.approximate_predict(clusterer, instance_image.view(-1,16).cpu())
            vis_path="output/scan6/objects_con/"
            print(np.max(all_labels))
            for obj_id in trange(np.max(all_labels)):
                path=os.path.join(vis_path,str(obj_id))
                mask=(all_labels==obj_id)
                object=get_obj_by_mask(gaussians,mask)
                for idx, viewpoint in enumerate(viewpoints):
                    render_pkg = render(viewpoint, object, pipe, background,True)
                    img=render_pkg["render"]
                    if mask.sum() < 200:
                        continue      
                    os.makedirs(path,exist_ok=True)
                    save_img_u8(img.permute(1,2,0).cpu().detach().numpy(),os.path.join(path,"RGB_"+str(idx)+".png"))
                    save_path=os.path.join(path,f'id{idx:05d}.png')
                    #saveRGB(label,save_path,class_n=-1)







