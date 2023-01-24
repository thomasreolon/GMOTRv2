# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils

from datasets.data_prefetcher import data_dict_to_cuda
from util.plot_utils import visualize_gt, train_visualize_pred

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, debug_out_path:str=False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('counting_loss', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    n_dl = len(data_loader)

    for d_i, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)

        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        c_loss = sum(outputs['count_loss'], 0)
        losses = losses + c_loss

        # reduce losses over all GPUs for logging purposes
        if d_i%print_freq==0:
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_scaled = {k: v  for k, v in loss_dict_reduced.items()
                                            if 'aux' not in k and '_1' not in k and '_2' not in k and '_3' not in k}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(counting_loss=c_loss.item() if isinstance(c_loss, torch.Tensor) else -1)

        # gather the stats from all processes
        if debug_out_path and d_i in {0,1,2,3,4,50%n_dl, 150%n_dl, 100%n_dl, n_dl//2, n_dl*4//5}:
            # utils.get_info()
            visualize_gt(data_dict, debug_out_path, d_i)
            train_visualize_pred(data_dict, outputs, debug_out_path, model.args.prob_detect, d_i)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
