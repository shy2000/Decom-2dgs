#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss,mask_mse,mask_cross_entropy,contrastive_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, MLP
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.render_utils import save_img_f32, save_img_u8
import numpy as np
import json
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def training_feature(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    with open(os.path.join(dataset.source_path,'images','instance_id.json'), 'r') as file:
        data = json.load(file)
    k=len(data)
    id_convert=torch.zeros(256,dtype=torch.int)#color to instance id
    for idx,i in enumerate(data.values()):
        id_convert[i]=idx
    id_convert[255]=k
    id_convert=id_convert.cuda()



    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        first_iter = 0
        gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_bg_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    classifier=MLP(3,64,k+1).cuda()
    class_optimizer=torch.optim.Adam(classifier.parameters(), lr=0.001)

    for iteration in range(first_iter, opt.iterations + 1):   

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt.include_feature)
        instance_image = render_pkg["instance_image"]

        if opt.contrastive: 
            instance_features=instance_image.permute(1, 2, 0).view(-1,16)
            temperature=100.0
            sample_num = opt.sample_num
            sam_mask=viewpoint_cam.sam_mask
            sam_mask=torch.from_numpy(sam_mask).view(-1).cuda()
            if len(sam_mask)!=len(instance_features):
                raise ValueError("mask size no match! {} {}".format(len(sam_mask),len(instance_features)))
            instance_features=instance_features[sam_mask!=0]
            sam_mask=sam_mask[sam_mask!=0]
            indices = torch.randperm(len(instance_features))[:sample_num].cuda()
            features=instance_features[indices]
            instance_labels=sam_mask[indices]
            main_loss=contrastive_loss(features,instance_labels,temperature)
      
        else:
            render_fea=True
            if render_fea :
                gt_image = viewpoint_cam.original_instance_image.cuda()
                #bg_mask=(gt_image[0,:] != 1.0)
                #bg=~bg_mask
                gt_label=gt_image[0,:]*255
                gt_label=id_convert[gt_label.int()].to(torch.int64).view(-1)
                #one_hot=torch.nn.functional.one_hot(gt_label,num_classes=k+1)
                label=classifier(instance_image.permute(1, 2, 0)).view(-1,k+1)
            else:
                pass#Render by id
            main_loss=torch.nn.functional.cross_entropy(label, gt_label)
        loss = main_loss 
        total_loss = loss 
        
        total_loss.backward()

        iter_end.record()

        if iteration%500==0:
            continue
            vis_path="output/vis/"
            save_img_u8(gt_image.permute(1,2,0).cpu().detach().numpy(),os.path.join(vis_path,str(int(iteration/500))+'img'+".png"))
            path=os.path.join(vis_path,str(int(iteration/500))+'id'+".png")
            saveRGB(label,path)


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, main_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background,opt.include_feature))
            if (iteration in saving_iterations):
                save_path=scene.model_path + "/chkpnt_contrastive_" + str(iteration) + ".pth"
                torch.save((gaussians.capture('Features'), iteration),save_path)
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                class_optimizer.step()
                class_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_with_feature" + str(iteration) + ".pth")
                torch.save(classifier, scene.model_path +"/classifier_chkpnt"+str(iteration)+'.pth')
                
        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer,include_feature=opt.include_feature)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None
    
    with torch.no_grad():
        if opt.contrastive:
            pass
        elif render_fea: 
            result_path="output/scan6/id_8/"
            os.makedirs(result_path,exist_ok=True)
            scene = Scene(dataset, gaussians,shuffle=False)
            viewpoint_stack = scene.getTrainCameras()
            for idx, viewpoint in enumerate(viewpoint_stack):
                render_pkg = render(viewpoint, gaussians, pipe, background, opt.include_feature)
                instance_image=render_pkg["instance_image"]
                label=classifier(instance_image.permute(1, 2, 0))
                path=os.path.join(result_path,f'id{idx:05d}.png')
                saveRGB(label,path)
            features=gaussians.get_instance_feature
            ids = classifier(features)
            ids = ids.argmax(dim=1, keepdim=True)
            gaussians.set_ids(ids.to(torch.int16))
            #print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #torch.save((gaussians.capture('Features'), iteration), scene.model_path + "/chkpnt_with_feature" + str(iteration) + ".pth")
            #torch.save(classifier, scene.model_path +"/classifier_chkpnt"+str(iteration)+'.pth')


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




def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)#TODO change renderArgs
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training_feature(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")