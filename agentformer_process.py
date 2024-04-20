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

def run_model(model, cfg):
    # Perform model1 operations using the input data
    epoch = 30
    split = 'test'
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')
    generator = data_generator(cfg, log, split=split, phase='testing')
    save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
    test_model(generator, save_dir, cfg, model, device, log)
    # flush the output to stdout
    sys.stdout.flush()
    # print "READY"
    print("READY", flush=True)


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