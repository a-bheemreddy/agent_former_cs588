import numpy as np
import argparse
import os
import sys
import subprocess
import shutil

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
import torch
import pandas as pd
import numpy as np
from test import test_model

# inspect results
gt_path = 'datasets/eth_ucy/inference_data/inference_data.txt'
real_path = 'results/inference/results/epoch_0005/test/samples/inference_data/frame_000007/sample_000.txt'
cfg = Config('inference')
device = 'cpu'
model = model_dict['dlow'](cfg)
model.set_device(device)
model.eval()

cp_path = cfg.model_path % 5
print(f'loading model from checkpoint: {cp_path}')
model_cp = torch.load(cp_path, map_location='cpu')
model.load_state_dict(model_cp['model_dict'], strict=False)
""" save results and compute metrics """
epoch = 30
split = 'test'
log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')
generator = data_generator(cfg, log, split=split, phase='testing')
save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
eval_dir = f'{save_dir}/samples'
test_model(generator, save_dir, cfg, model, device, log)

# log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
# cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
# subprocess.run(cmd.split(' '))
def run_model():
    generator = data_generator(cfg, log, split=split, phase='testing')
    data = generator()
    if data is None:
        return
    seq_name, frame = data['seq'], data['frame']

    frame = int(frame)
    sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
    sys.stdout.flush()

    gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
    with torch.no_grad():
        recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
    recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

    """save samples"""
    recon_dir = os.path.join(save_dir, 'recon'); mkdir_if_missing(recon_dir)
    sample_dir = os.path.join(save_dir, 'samples'); mkdir_if_missing(sample_dir)
    gt_dir = os.path.join(save_dir, 'gt'); mkdir_if_missing(gt_dir)
    for i in range(sample_motion_3D.shape[0]):
        save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
    save_prediction(recon_motion_3D, data, '', recon_dir)        # save recon
    num_pred = save_prediction(gt_motion_3D, data, '', gt_dir)              # save gt
    total_num_pred += num_pred

run_model()