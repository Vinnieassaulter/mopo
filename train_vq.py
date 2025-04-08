import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader, random_split

from models.vq.model import VQVAE
from models.vq.vq_trainer import VQTokenizerTrainer
from options.vq_options import arg_parse
from data.mopo_dataset import MarkerDataset
from utils import paramUtil
import numpy as np

# from models.t2m_eval_wrapper import EvaluatorModelWrapper
# from utils.get_opt import get_opt
# from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from utils.fixseed import fixseed


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    # opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    # os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "soma":
        opt.data_root = './mocap_data/V48_01_SOMA'
        opt.marker_dir = pjoin(opt.data_root, 'marker_dataset')
        opt.noise_dir = pjoin(opt.data_root, 'amass_marker_noise')
        opt.n_max_markers = 53
        # kinematic_chain = paramUtil.t2m_kinematic_chain
        # dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    # wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    # eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    train_marker_path = pjoin(opt.marker_dir, 'train/markers.pt')
    train_noise_path = pjoin(opt.noise_dir, 'train/amass_marker_noise_model.npz')
    # train_pose_path = pjoin(opt.marker_dir, 'train/pose_body.pt')

    val_marker_path = pjoin(opt.marker_dir, 'vald/markers.pt')
    val_noise_path = pjoin(opt.noise_dir, 'vald/amass_marker_noise_model.npz')
    # val_pose_path = pjoin(opt.marker_dir, 'val/pose_body.pt')

    # 修改
    net = VQVAE(
                input_dim=3, 
                hidden_dim=64, 
                latent_dim=32, 
                num_embeddings=512, 
                num_markers=53
    )
    

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    # print("Total parameters of discriminator net: {}".format(pc_vq))
    # all_params += pc_vq_dis

    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))

    trainer = VQTokenizerTrainer(opt, vq_model=net)
    # dataset = MarkerDataset(opt, train_marker_path, train_noise_path, train_pose_path,
    #                         n_max_markers=opt.n_max_markers, n_noised_frames=20, n_max_noised_markers=20,
    #                         window_size=opt.window_size)

    train_dataset = MarkerDataset(train_marker_path, train_noise_path,  
                                  n_max_markers=opt.n_max_markers, n_noised_frames=20, n_max_noised_markers=5, window_size=opt.window_size)
    val_dataset = MarkerDataset(val_marker_path, val_noise_path, 
                                n_max_markers=opt.n_max_markers, n_noised_frames=20, n_max_noised_markers=5, window_size=opt.window_size)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)
    
    # eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)
    # trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_t2m)

    trainer.train(train_loader, val_loader, None, None, None)

## train_vq.py --dataset_name kit --batch_size 512 --name VQVAE_dp2 --gpu_id 3
## train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp2_b256 --gpu_id 2
## train_vq.py --dataset_name kit --batch_size 1024 --name VQVAE_dp2_b1024 --gpu_id 1
## python train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp1_b256 --gpu_id 2