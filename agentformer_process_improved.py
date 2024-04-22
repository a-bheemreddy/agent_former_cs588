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
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from dataloader_improved import data_generator
import torch
import pandas as pd
import numpy as np


def get_model_prediction(data, sample_k, model):
    model.set_data(data)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return sample_motion_3D

def run_model(model, cfg, file_path):
    gt_data = np.genfromtxt(file_path, delimiter=' ', dtype=str)
    best_samples = [0,1,17,18,19]
    # Perform model1 operations using the input data
    epoch = 30
    split = 'test'
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')
    generator = data_generator(cfg, gt_data, log, split=split, phase='testing')
    save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
    sample_motion_3D = run_model_on_data(generator, save_dir, cfg, model, device, log) # [samples, num_agents, future_frames, 2]
    sample_motion_3D = sample_motion_3D[best_samples]
    # write to file or return  
    # select the
    # flush the output to stdout
    sys.stdout.flush()
    # print "READY"
    print("RAN MODEL !!!!!", flush=True)
    print(sample_motion_3D.shape)

def run_model_on_data(generator, save_dir, cfg, model, device, log):
    total_num_pred = 0
    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        seq_name, frame = data['seq'], data['frame']

        frame = int(frame)
        
        with torch.no_grad():
            sample_motion_3D = get_model_prediction(data, cfg.sample_k, model)
        sample_motion_3D = sample_motion_3D * cfg.traj_scale
    
        return sample_motion_3D
        

if __name__ == "__main__":
    cfg = Config('inference')
    device = 'cuda'

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

    file_path = 'datasets/eth_ucy/inference_data/inference_data.txt'
    run_model(model, cfg, file_path)

    # while True:
    #     # for line in sys.stdin:
    #     #     input_data = line.strip()
    #     #     if not input_data:
    #     #         break
    #     logger.info('Starting up')
    #     input_data = sys.stdin.readline().strip()
    #     logger.info(f'Have input data {input_data}')
    #     #print(input_data)
    #     if not input_data:
    #         logger.info('ERROR BREAKNG BREAKING')
    #         break
    #     run_model(model, cfg)
    #     logger.info(f'Have output data Finished running model')
    #     sys.stdout.flush()