# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import datetime
import random
import time
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.tool import load_model
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_mot
from models import build_model
from configs.defaults import get_args_parser

from tqdm import tqdm

def main(args):

    dataset_train = build_dataset(image_set='train', args=args)

    for i in tqdm(range(len(dataset_train))):
        print('.')
        data = dataset_train[i]

        print([x.shape for x in data['imgs']],  [(x.boxes.min(),x.boxes.max()) for x in data['gt_instances']], [(len(x.obj_ids),len(set(x.obj_ids.tolist()))) for x in data['gt_instances']], data['exemplar'][0].shape, data['exemplar'][1].tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
