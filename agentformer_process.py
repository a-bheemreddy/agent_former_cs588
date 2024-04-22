# model1.py
import sys
import logging
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

# How many of the top trajectories that we want to keep
NUM_TOP_TRAJECTORIES = 3
# 8 frames used to base future trajectories off of (current frame plus previous 7)
NUM_PREV_FRAMES = 7
NUM_FUTURE_FRAMES = 12

def save_best_trajectories(sample_motions, cfg, trajs_to_save=NUM_TOP_TRAJECTORIES):
    # sample motions: num_frames_recorded x NUM_TOP_TRAJECTORIES x num_pedestrians x 12 x 2
    # (first two layers of sample_motions are lists)
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

def run_model(model, cfg):
    # Perform model1 operations using the input data
    epoch = 30
    split = 'test'
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')
    generator = data_generator(cfg, log, split=split, phase='testing')
    save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
    sample_motions = test_model(generator, save_dir, cfg, model, device, log)
    save_best_trajectories(sample_motions, cfg)
    # flush the output to stdout
    sys.stdout.flush()
    # print "READY"
    print("READY", flush=True)


if __name__ == "__main__":
    cfg = Config('inference')
    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_dict['dlow'](cfg)
    model.set_device(device)
    model.eval()

    cp_path = cfg.model_path % 5
    print(f'loading model from checkpoint: {cp_path}')
    model_cp = torch.load(cp_path, map_location='cpu')
    model.load_state_dict(model_cp['model_dict'], strict=False)
    # print("we here")
    # create logger for file logging.txt
    logger = logging.getLogger('logging')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('logging.txt')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    # Log some messages


    while True:
        # for line in sys.stdin:
        #     input_data = line.strip()
        #     if not input_data:
        #         break
        logger.info('Starting up')
        input_data = sys.stdin.readline().strip()
        logger.info(f'Have input data {input_data}')
        #print(input_data)
        if not input_data:
            logger.info('ERROR BREAKNG BREAKING')
            break
        run_model(model, cfg)
        logger.info(f'Have output data Finished running model')
        sys.stdout.flush()