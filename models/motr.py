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
DETR model and criterion classes.
"""
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import List
import cv2, numpy as np

from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .transformers import build_deforamble_transformer
from .qim import pos2posemb, build as build_query_interaction_layer
from .detr_loss import SetCriterion, MLP, sigmoid_focal_loss
from .bmn.models import build_model as build_bmn

class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.g_blur = None

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses


    def loss_count(self, outputs, gt_inst):
        boxes = gt_inst.boxes
        corr_map = outputs['corr'].view(*self.corr_hw)      #H,W
        density_map = outputs['dmap'][0,0]   #H,W

        # density loss
        count_map = torch.zeros_like(density_map)
        std = 20 # boxes[:,2:].mean().item() * min(count_map.shape)
        blurrer = self.get_gauss_blur(std).to(count_map.device)
        for bb in boxes:
            x, y = int(bb[0]*density_map.shape[1]), int(bb[1]*density_map.shape[0])
            count_map[y,x] = 1.
        with torch.no_grad():
            count_map = blurrer(count_map[None,None])[0,0]
            count_map = count_map - count_map.min()
            count_map = count_map / count_map.sum() * len(boxes)
        density_loss = F.mse_loss(density_map, count_map)
        
        # if torch.rand(1)>0.5:
        #     print('ok')
        #     tmp = torch.cat((density_map, count_map), dim=1).clone().detach().cpu().numpy()[:,:,None]
        #     tmp = (tmp-tmp.min()) / (tmp.max()-tmp.min())
        #     cv2.imshow('loss', tmp)
        #     cv2.waitKey(40)

        # correlation loss
        count_map = F.adaptive_avg_pool2d(count_map[None], self.corr_hw).view(self.corr_hw)
        positive = count_map > count_map.mean()+count_map.std()
        corr_map = (corr_map-corr_map.mean()) / (corr_map.std()+1e-8)
        corr_map = corr_map.exp()
        corr_loss = - ( corr_map[positive].sum() / (corr_map.sum()+1e-10) ).log()
        # corr_loss = 0

        return corr_loss*1e-5, density_loss


    def get_gauss_blur(self, std=14, kernel_size=32):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        if self.g_blur is None:
            mean = (kernel_size - 1)/2.

            # Calculate the 2-dimensional gaussian kernel which is
            gaussian_kernel = (1./(2.*math.pi*std)) *\
                            torch.exp(
                                -torch.sum((xy_grid - mean)**2., dim=-1) /\
                                (2*std)
                            )
            # Make sure sum of values in gaussian kernel equals 1.
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

            gaussian_filter = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                        kernel_size=kernel_size, padding='same', bias=False)

            gaussian_filter.weight.data = gaussian_kernel
            gaussian_filter.weight.requires_grad = False
            self.g_blur = gaussian_filter.float()
        return self.g_blur


    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        i, j = torch.where(track_instances.obj_idxes[:, None] == obj_idxes)
        track_instances.matched_gt_idxes[i] = j

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i), device=pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i), device=pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})

        if 'ps_outputs' in outputs:
            for i, ps_outputs in enumerate(outputs['ps_outputs']):
                if ps_outputs['pred_boxes'].numel() == 0: continue
                ar = torch.arange(len(gt_instances_i), device=obj_idxes.device)
                l_dict = self.get_loss('boxes',
                                        ps_outputs,
                                        gt_instances=[gt_instances_i],
                                        indices=[(ar, ar)],
                                        num_boxes=1, )
                self.losses_dict.update(
                    {'frame_{}_ps{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                        l_dict.items()})
        
        if outputs['corr'] is not None:
            cr_loss, cn_loss = self.loss_count(outputs, gt_instances_i)
            self.losses_dict.update({
                f'corr_loss_frame_{self._current_frame_idx}':cr_loss, 
                f'count_loss_frame_{self._current_frame_idx}':cn_loss
            })

        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses



class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        # prob = out_logits.sigmoid()
        scores = out_logits[..., 0].sigmoid()
        # scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = torch.full_like(scores, 0)
        # track_instances.remove('pred_logits')
        # track_instances.remove('pred_boxes')
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR(nn.Module):
    def __init__(self, backbone, transformer, track_embed, args, criterion, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.args = args
        q_per_row = int(math.sqrt(args.num_queries))
        args.num_queries = q_per_row**2
        self.num_queries = args.num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = args.num_feature_levels
        self.use_checkpoint = args.use_checkpoint
        self.query_denoise = args.query_denoise

        emb = (1, args.num_queries) if self.args.use_bmn else (args.num_queries, -1)
        grid = torch.stack(torch.meshgrid(.5+torch.arange(q_per_row)/q_per_row, .5+torch.arange(q_per_row)/q_per_row),dim=2).view(-1,2)
        self.q_refs = nn.Parameter(torch.cat((grid, .02+.1*torch.rand(args.num_queries, 2)), dim=1))
        self.q_emb = nn.Parameter(torch.rand(emb[0], hidden_dim).expand(emb[1], -1))
        self.prop_embed = nn.Parameter(torch.rand(1, hidden_dim))
        if args.query_denoise:
            self.refine_embed = nn.Parameter(torch.rand(1, hidden_dim))

        if self.args.use_bmn:
            self.bmn = build_bmn(args.bmn_pretrained)
        if args.num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(args.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = args.aux_loss
        self.with_box_refine = args.with_box_refine
        self.two_stage = args.two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if args.two_stage else transformer.decoder.num_layers
        if args.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if args.two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase(args.prob_detect, args.prob_detect*.8)
        self.criterion = criterion
        self.memory_bank = args.memory_bank_type
        self.mem_bank_len = 0 if args.memory_bank_type is None else self.memory_bank.max_his_length

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        _, d_model = self.q_emb.shape  # (300, 512)
        device = self.q_emb.device

        track_instances.ref_pts = self.q_refs
        track_instances.query_pos = self.q_emb

        track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, exemplar, track_instances: Instances, gtboxes=None):
        ## Extract Features from Frame
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        ## Additional Feats Levels
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        ## Extract Features from Exemplar and add as feature layer
        exefeatures=self._extract_exemplar_features(exemplar, srcs)
        srcs.append(exefeatures)
        masks.append(torch.zeros_like(exefeatures[:,0]).bool())
        pos.append(torch.zeros_like(exefeatures))


        ## prepare input for transformer
        query_embed = track_instances.query_pos
        ref_pts = track_instances.ref_pts
        attn_mask = None

        # proposals from agnostic counting
        proposed, dmap, corr = self._get_proposals(samples.tensors, exemplar.tensors)
        if proposed is not None:
            pr_tgt = self.prop_embed.expand(proposed.size(0), -1)
            query_embed = torch.cat([pr_tgt, query_embed])
            ref_pts = torch.cat([proposed, ref_pts])

        # Add GT to guide Learning
        if gtboxes is not None:
            n_dt = len(query_embed)
            ps_tgt = self.refine_embed.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([query_embed, ps_tgt])
            ref_pts = torch.cat([ref_pts, gtboxes])
            attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=bool, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = True


        ## TRANSFORMER
        hs, init_reference, inter_references, _, _ = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts, attn_mask=attn_mask)
        
        # remove proposed
        if proposed is not None:
            n_pr = proposed.size(0)
            hs = hs[..., n_pr:, :]
            init_reference = init_reference[..., n_pr:, :]
            inter_references = inter_references[..., n_pr:, :]

        ## Head
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        ## Outputs Dict
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['dmap'] = dmap
        out['corr'] = corr
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out

    def _extract_exemplar_features(self, exemplar, srcs):
        exefeatures = []

        if False and self.args.extract_exe_from_img:
            for src in srcs:
                b,c,H,W = src.shape
                bb = (exemplar.view(2,2) * torch.tensor([W,H], device=exemplar.device).view(1,2)).int().flatten()  # coords in src
                wc,hc,w,h = bb
                hh=lambda h: max(0,min(h,H-1)) ; ww=lambda w: max(0,min(w,W-1))
                exefeatures.append(src[:,:, hh(hc-h//2): hc+h//2+2, ww(wc-w//2):wc+w//2+2].mean(dim=(2,3)))
                exefeatures.append((src[:,:,hh(hc-h//4),wc]+src[:,:,hh(hc-h//4), wc]+src[:,:,hc, ww(wc-w//4)]+src[:,:,hc, ww(wc+w//4)])/4)
                exefeatures.append(src[:,:,hc,wc])
        else:
            ## Extract Features from Exemplar and add as feature layer
            exefeatures=[]
            for l, (feat, _) in enumerate(zip(*self.backbone(exemplar))):
                esrc, mask = feat.decompose()
                esrc = self.input_proj[l](esrc)
                p = (mask.sum() / mask.numel())
                b,c,h,w = esrc.shape
                hc,wc = h//2,w//2
                exefeatures.append(esrc[:,:,hc-int(p*hc):hc+int(p*hc)+2, wc-int(p*wc):wc+int(p*wc)+2].mean(dim=(2,3)))
                exefeatures.append((esrc[:,:,hc-int(.5*p*hc), wc]+esrc[:,:,hc-int(.5*p*hc), wc]+esrc[:,:,hc, wc-int(.5*p*wc)]+esrc[:,:,hc, wc+int(.5*p*wc)])/4)
                exefeatures.append(esrc[:,:,hc,wc])
        exefeatures = torch.stack(exefeatures, dim=-1).view(b,c,-1,3)
        return exefeatures

    def _get_proposals(self, image, exemplar):
        if self.args.use_bmn==0: return None,None,None
        device = image.device
        # resize image
        w,h = 512, 512*image.shape[2]//image.shape[3]
        r = w / image.shape[2]
        image = F.adaptive_avg_pool2d(image, (h,w))

        we,he = int(exemplar.shape[3]*r), int(exemplar.shape[2]*r)
        exemplar =F.adaptive_avg_pool2d(exemplar, (he,we))

        # calculate scale
        scale = exemplar.shape[3] / image.shape[3] * 0.5 + exemplar.shape[2] / image.shape[3] * 0.5 
        scale = scale // (0.5 / 20)
        scale = scale if scale < 20 - 1 else 20 - 1
        patch = {'scale_embedding':torch.tensor([[scale]], device=device).int(), 'patches':exemplar[:,None]}

        # get proposals
        res = self.bmn(image, patch, True)
        interest = F.adaptive_avg_pool2d(res['density_map'], image.shape[-2:]) [0,0] #HW
        self.criterion.corr_hw = res['f_shape']

        # generate q_refs from proposals
        good_pixels1 = torch.zeros_like(interest).bool()
        c1 = interest[:-1, :] >= interest[1: , :]  # bigger than up
        c2 = interest[1: , :] >  interest[:-1, :]  # bigger than down
        c3 = interest[:, :-1] >= interest[:, 1: ]  # bigger than left
        c4 = interest[:, 1: ] >  interest[:, :-1]  # bigger than right

        good_pixels1[1:-1, 1:-1] = c1[1:, 1:-1] & c2[:-1, 1:-1] & c3[1:-1, 1:] & c4[1:-1, :-1]
        good_pixels2 = interest > interest.max()/2

        # print(interest.mean(), interest.std(), good_pixels1.sum(), good_pixels2.sum())
        if not (good_pixels1 & good_pixels2).any(): 
            return None, res['density_map'], res['corr_map']
        good_pixels = (good_pixels1 & good_pixels2).nonzero(as_tuple=True)
        xy = torch.stack((good_pixels[-1], good_pixels[-2]), dim=1).float()
        xy = xy / torch.tensor([w,h],device=device).view(1,2)

        bb = torch.tensor([[exemplar.shape[3]/image.shape[3],  exemplar.shape[2]/image.shape[2]]], device=device).expand(xy.shape[0], -1)


        if self.args.debug and torch.rand(1)>0.7:
            tmp = image.cpu()[0].permute(1,2,0).numpy()[:,:,::-1]
            tmp = (tmp-tmp.min()) / (tmp.max()-tmp.min())

            for coord in xy:
                x = int(coord[0]*tmp.shape[1]) 
                y = int(coord[1]*tmp.shape[0])
                tmp[y,x] = (0,0,1.)
                tmp[y+1,x+1] = (0,0,1.)
                tmp[y-1,x+1] = (0,0,1.)
                tmp[y-1,x-1] = (0,0,1.)
                tmp[y+1,x-1] = (0,0,1.)
                tmp[y+2,x+2] = (1,0,.3)
                tmp[y-2,x+2] = (1,0,.3)
                tmp[y-2,x-2] = (1,0,.3)
                tmp[y+2,x-2] = (1,0,.3)
            tmp2 = interest.clone().detach().cpu()[:,:,None].expand(-1,-1,3).numpy()
            tmp2 = (tmp2-tmp2.min()) / (tmp2.max()-tmp2.min())
            tmp = np.concatenate((tmp,tmp2), axis=1)
            # cv2.imshow('ddd', tmp)

            cv2.imwrite(f'{self.args.output_dir}/debug/BMN_{len(xy)//10}.jpg', np.uint8(tmp*255))


        return torch.cat((xy,bb),dim=1), res['density_map'], res['corr_map']   # Nx4


    def _post_process_single_image(self, frame_res, track_instances, is_last):
        if self.query_denoise > 0:
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            ps_boxes = frame_res['pred_boxes'][:, n_ins:]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res['aux_outputs']:
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs

        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        if self.training:
            if self.args.debug:
                dt_instances = track_instances.clone()
                self.track_base.update(dt_instances)
                frame_res['dt_instances'] = dt_instances[dt_instances.obj_idxes>=0]
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)

        # improve queries (before it was skipped if it was the last)
        frame_res['track_instances'] = self.track_embed({'track_instances':track_instances})

        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, exemplar=None, exe_bb=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list([img])
        if not isinstance(exemplar, NestedTensor):
            exemplar = nested_tensor_from_tensor_list([exemplar])
        else:
            exemplar = exe_bb

        if track_instances is None:
            track_instances = self._generate_empty_tracks()
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(),
                track_instances])
        res = self._forward_single_image(img, exemplar,
                                         track_instances=track_instances)
        res = self._post_process_single_image(res, track_instances, True)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        return track_instances

    def forward(self, data: dict, track_instances=None):
        # print('-------------------')
        # print(torch.cat([x.view( -1) for x in self.bmn.refiner.parameters() ] ).mean())
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        exemplar = data['exemplar'][1] if self.args.extract_exe_from_img else data['exemplar'][0]

        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'post_proc': [],
        }
        keys = list(self._generate_empty_tracks()._fields.keys())
        for frame_index, (frame, gt) in enumerate(zip(frames, data['gt_instances'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1

            if self.query_denoise > 0 and gt is not None and self.training:
                l_1 = l_2 = self.query_denoise
                gtboxes = gt.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            if track_instances is None:
                track_instances = self._generate_empty_tracks()
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(),
                    track_instances])

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, exemplar, gtboxes, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    exemplar = nested_tensor_from_tensor_list([exemplar])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, exemplar, tmp, gtboxes)
                    closs = (frame_res['dmap'], frame_res['corr']) if frame_res['dmap'] is not None else ()
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                    ) + closs

                args = [frame, exemplar, gtboxes] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                n_dec = self.transformer.decoder.num_layers - 1
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'hs': tmp[2],
                    'aux_outputs': [{
                        'pred_logits': tmp[3+i],
                        'pred_boxes': tmp[3+n_dec+i],
                    } for i in range(n_dec)],
                    'dmap':tmp[-2] if len(tmp)>3+n_dec*2 else None, 
                    'corr':tmp[-1] if len(tmp)>3+n_dec*2 else None
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                exemplar = nested_tensor_from_tensor_list([exemplar])
                frame_res = self._forward_single_image(frame, exemplar, track_instances, gtboxes)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last) # TODO do it inside of checkpoint

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

            inst = frame_res['dt_instances'] if self.training and self.args.debug else track_instances.clone()
            dt_instances = self.post_process(inst.to('cpu'), (data['imgs'][0].shape[-2:]))
            outputs['post_proc'].append(dt_instances)

            if self.args.debug and torch.rand(1)>0.8:     # if true will show HUNGARIAN MATCHED detections for each image (debugging)
                dt_instances = self.post_process(track_instances, data['imgs'][0].shape[-2:])

                keep = dt_instances.scores > .02
                keep &= dt_instances.obj_idxes >= 0
                dt_instances = dt_instances[keep]

                wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
                areas = wh[:, 0] * wh[:, 1]
                keep = areas > 100
                dt_instances = dt_instances[keep]

                if len(dt_instances)>0:
                    bbox_xyxy = dt_instances.boxes.tolist()
                    identities = dt_instances.obj_idxes.tolist()

                    img = data['imgs'][frame_index].clone().cpu().permute(1,2,0).numpy()[:,:,::-1]
                    # img = (img-img.min())/(img.max()-img.min())
                    img = img/4 +.4
                    for xyxy, track_id in zip(bbox_xyxy, identities):
                        if track_id < 0 or track_id is None:
                            continue
                        x1, y1, x2, y2 = [int(a) for a in xyxy]
                        color = tuple([(((5+track_id*3)*4909 % p)%256) /256 for p in (3001, 1109, 2027)])

                        tmp = img[ y1:y2, x1:x2].copy()
                        img[y1-3:y2+3, x1-3:x2+3] = color
                        img[y1:y2, x1:x2] = tmp
                    cv2.imwrite(f'{self.args.output_dir}/debug/match_{len(dt_instances)//10}.jpg', np.uint8(img*255))

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    num_classes = 1
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'corr_loss_frame_{}'.format(i): args.cls_loss_coef,
                            'count_loss_frame_{}'.format(i): args.cls_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
            for j in range(args.dec_layers):
                weight_dict.update({"frame_{}_ps{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_ps{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_ps{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
    args.memory_bank_type = None
    losses = ['labels', 'boxes']
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = MOTR(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        args=args,
        criterion=criterion,
        num_classes=num_classes
    )
    return model, criterion, postprocessors
