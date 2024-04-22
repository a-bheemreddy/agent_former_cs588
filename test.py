import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
import json

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing

# How many of the top trajectories that we want to keep
NUM_TOP_TRAJECTORIES = 3
# 8 frames used to base future trajectories off of (current frame plus previous 7)
NUM_PREV_FRAMES = 7
NUM_FUTURE_FRAMES = 12
TOP_PERFORMING_INDICES = [0, 18, 19]

def get_model_prediction(data, sample_k, model):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D


def save_prediction(pred, data, suffix, save_dir, cfg):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = pred[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num

def test_model(generator, save_dir, cfg, model, device, log):
    total_num_pred = 0
    sample_motions = []
    while not generator.is_epoch_end():
        current_sample_motion = []
        data = generator()
        if data is None:
            continue
        seq_name, frame = data['seq'], data['frame']

        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        sys.stdout.flush()
        
        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k, model)
        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

        """save samples"""
        recon_dir = os.path.join(save_dir, 'recon'); mkdir_if_missing(recon_dir)
        sample_dir = os.path.join(save_dir, 'samples'); mkdir_if_missing(sample_dir)
        gt_dir = os.path.join(save_dir, 'gt'); mkdir_if_missing(gt_dir)
        for i in range(sample_motion_3D.shape[0]): # 20 x num_peds x 12 x 2
            save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir, cfg)
            if i in TOP_PERFORMING_INDICES:
                current_sample_motion.append(sample_motion_3D[i])
        save_prediction(recon_motion_3D, data, '', recon_dir, cfg)        # save recon
        num_pred = save_prediction(gt_motion_3D, data, '', gt_dir, cfg)              # save gt
        total_num_pred += num_pred
        sample_motions.append(current_sample_motion)

    print_log(f'\n\n total_num_pred: {total_num_pred}', log)
    if cfg.dataset == 'nuscenes_pred':
        scene_num = {
            'train': 32186,
            'val': 8560,
            'test': 9041
        }
        #assert total_num_pred == scene_num[generator.split]
    return sample_motions

def save_best_trajectories(sample_motions, cfg, trajs_to_save=NUM_TOP_TRAJECTORIES):
    # sample motions: num_frames_recorded x NUM_TOP_TRAJECTORIES x num_pedestrians x 12 x 2
    # (first two layers of sample_motions are lists)
    print(len(sample_motions), len(sample_motions[0]))
    top_results_dir = os.path.join(cfg.result_dir, 'top_paths'); mkdir_if_missing(top_results_dir)
    for frame in range(len(sample_motions)):
        current_frame = frame + NUM_PREV_FRAMES
        for traj in range(NUM_TOP_TRAJECTORIES):
            current_file = f'{top_results_dir}/frame_{int(current_frame):06d}/sample_{traj:03d}.txt'
            mkdir_if_missing(current_file)
            num_pedestrians = sample_motions[frame][traj].shape[0]

            with open(current_file, 'w+') as f:
                # Pedestrian IDs are 1-indexed in AgentFormer
                for p_id in range(1, num_pedestrians + 1):
                    for future_frame in range(NUM_FUTURE_FRAMES):
                        current_ped_x = sample_motions[frame][traj][p_id - 1][future_frame][0]
                        current_ped_y = sample_motions[frame][traj][p_id - 1][future_frame][1]
                        line = f"{float(current_frame + future_frame + 1)} {float(p_id)} {float(current_ped_x)} {float(current_ped_y)}\n"
                        f.write(line)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = [args.data_eval]

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples'
            if not args.cached:
                sample_motions = test_model(generator, save_dir, cfg, model, device, log)
                save_best_trajectories(sample_motions, cfg)

            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
            subprocess.run(cmd.split(' '))

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)