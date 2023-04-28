import os
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from ._transforms import make_synth_transforms
from models.structures import Instances


def build(split, args):
    # Get base path
    root = f'{args.synth_path}/{split}/'
    assert os.path.isdir(root), f'provided MOT path {root} does not exist'

    # Get data autmentations 
    transform = make_synth_transforms(args, split)

    return SynthDataset(root, args, transform)


class SynthDataset(Dataset):
    def __init__(self, root, args, transform=None):
        self.root = root
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = args.sampler_lengths # frames per video interval

        # Load data
        self.annotations, self.indices = load_files(root, args.sampler_lengths, args.fast)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, fail=1):
        # Get data
        vid, f_index = self.indices[idx]
        indices = [f_index + i for i in range(self.num_frames_per_batch)]
        images, targets = self.pre_continuous_frames(vid, indices)

        # Apply data augmentations
        if self.transform is not None:
            images, targets = self.transform(images, targets)

        # Convert to instances
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)

        # Get exemplar
        exemplar = self.get_exemplar(images[0], targets[0])
        if len(exemplar)==0:
            if fail: return self.__getitem__(idx+1, fail=fail-1)
            else:    raise Exception('No exemplar found')

        # Return
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'exemplar': exemplar,
        }


    def get_exemplar(self,img,target):
        exemplars = []
        for bb in target['boxes']:
            bb = bb.clone()
            bb = (bb.view(2,2) * torch.tensor([img.shape[2],img.shape[1]]).view(1,2)).flatten()  # coords in img
            bb = torch.cat((bb[:2]-bb[2:]/2, bb[:2]+bb[2:]/2)).int()               # x1y1x2y2
            patch = img[:, bb[1]:bb[3], bb[0]:bb[2]]
            exemplars.append(patch)
        exemplars = [x for x in exemplars if max(*x.shape[1:])>32 and max(*x.shape[1:])/(min(*x.shape[1:])+0.02)<6]
        return exemplars

    def _pre_single_frame(self, vid, idx):
        # Load image
        img_path = f'{self.root}/GenericMOT_JPEG_Sequence/{vid}/img1/{idx:06d}.jpg'
        img = Image.open(img_path)

        # Load targets
        xywhi = self.annotations[vid][idx]
        target = {
            'boxes': [box[:4] for box in xywhi],
            'labels': [0 for _ in range(len(xywhi))],
            'obj_ids': [id[-1] for id in xywhi],
        }
        target['labels'] = torch.tensor(target['labels'])
        target['obj_ids'] = torch.tensor(target['obj_ids'], dtype=torch.float64)
        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32).reshape(-1, 4)   
        
        return  img, target
        
    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])




# load tracking labels (MOT format)
def load_labels(txt_path):
    gt = defaultdict(lambda: [])
    data = np.genfromtxt(txt_path, delimiter=',', dtype=np.int64)
    data = data.reshape(-1, 10)

    for line in data:
        gt[line[0]].append( [line[2], line[3], line[2]+line[4], line[3]+line[5], line[1]] )
    return gt

def load_files(dataset_path, num_frames_per_batch, fast=False):
    videos = os.listdir(f'{dataset_path}/GenericMOT_JPEG_Sequence')
    annotations = {}
    indices_videoframe = []

    n = 100 if fast else int(1e9)
    for vid in videos[:n]:
        # Load bounding boxes
        annotations_path = f'{dataset_path}/track_label/{vid}.txt'
        annotations[vid] = load_labels(annotations_path)
        annotations[vid] = {k:v for k,v in annotations[vid].items() if k<n/10}

        # get possible start frames
        t_min = min(annotations[vid].keys())
        t_max = max(annotations[vid].keys()) + 1
        for t in range(t_min, t_max - num_frames_per_batch):
            indices_videoframe.append((vid, t))
    return annotations, indices_videoframe

def targets_to_instances(targets: dict, img_shape) -> Instances:
    gt_instances = Instances(tuple(img_shape))
    print('ok')
    gt_instances.boxes = targets['boxes']
    gt_instances.labels = targets['labels']
    gt_instances.obj_ids = targets['obj_ids']
    return gt_instances