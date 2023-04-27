# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .dance import build as build_e2e_dance
from .joint import build as build_e2e_joint
from .fscd import build as build_e2e_fscd
from .gmot import build as build_e2e_gmot
from .joint2 import build as build_joint
from .coco import build as build_e2e_coco

def build_dataset(image_set, args):
    if args.dataset_file == 'e2e_joint':
        return build_e2e_joint(image_set, args)
    if args.dataset_file == 'e2e_dance':
        return build_e2e_dance(image_set, args)
    if args.dataset_file == 'e2e_fscd':
        return build_e2e_fscd(image_set, args)
    if args.dataset_file == 'e2e_gmot':
        return build_e2e_gmot(image_set, args)
    if args.dataset_file == 'e2e_coco':
        return build_e2e_coco(image_set, args)
    if 'joint' in args.dataset_file:
        return build_joint(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
