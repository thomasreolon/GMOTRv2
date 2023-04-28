from collections import defaultdict
from PIL import Image
import pathlib
import json

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from ._transforms import make_synth_transforms
from models.structures import Instances


class FSCDataset(Dataset):
    def __init__(self, args, split, transform) -> None:
        super().__init__()
        self.args = args
        self.transform = transform

        # folder with images
        self.path_imgs = args.fscd_path+'/images_384_VarV2/'

        # get data from json
        det2 = json.load(open(args.fscd_path+'/instances_test.json', 'r'))  
        det = json.load(open(args.fscd_path+'/instances_val.json', 'r')) 
        all_ann = json.load(open(args.fscd_path+'/annotation_FSC147_384.json', 'r'))

        # load listof(images,detections)
        self.detections = self.load_anns(all_ann, det) + self.load_anns(all_ann, det2)

    def load_anns(self, d_exe, d_det):
        ## TODO -> use fewshot counting exemplar...
        bbs = defaultdict(lambda: list())
        id2img = {imginfo['id']:imginfo['file_name']   for imginfo in d_det['images']}

        for boxinfo in d_det['annotations']:
            if boxinfo['image_id'] not in id2img: 
                continue
            img_path = id2img[ boxinfo['image_id'] ]
            b = d_exe[img_path]['box_examples_coordinates'][0]
            b = b[0][1], b[0][0] , b[2][1], b[2][0]
            bbs[img_path].append(boxinfo['bbox'])
        bbs = [(k,v,b) for k,v in bbs.items() if len(v)<250]
        return bbs

    def __len__(self):
        return len(self.detections)

    def _pre_single_frame(self, idx):
        img_path, bbs, e_bb = self.detections[idx]
        img = Image.open(self.path_imgs+img_path)

        obj_idx_offset = idx * 1000
        target = {
            'boxes': bbs,
            'labels': [0 for _ in range(len(bbs))],
            'obj_ids': [obj_idx_offset+id for id in range(len(bbs))],

            # could be removed if fix transformation
            'iscrowd': torch.tensor([0 for _ in range(len(bbs))]),
            'scores': torch.tensor([0 for _ in range(len(bbs))]),
        }
        target['labels'] = torch.tensor(target['labels'])
        target['obj_ids'] = torch.tensor(target['obj_ids'], dtype=torch.float64)
        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32).reshape(-1, 4)
        target['boxes'][:,2:] += target['boxes'][:,:2]
        
        # get exemplar
        exe = None #F.normalize(F.to_tensor(img.crop(e_bb)), (0.485,0.456,0.406),(0.229,0.224,0.225))


        return [img], [target], [exe]

    def __getitem__(self, idx, failed=False):
        idx = idx%len(self.detections)
        images, targets, exemplar = self._pre_single_frame(idx)
        if self.transform is not None:
            images, targets = self.transform(images, targets)
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)

        if any([ t['boxes'].shape[0]==0 for t in targets ]):
            if failed: idx = idx+53
            return self.__getitem__(idx, True)

        exemplar = self.get_exemplar(images[0], targets[0])

        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'exemplar': exemplar,
        }


    def get_exemplar(self,img,target):
        bbnorm = target['boxes'].clone().view(-1,2,2).clip(0,1)
        bb = (bbnorm * torch.tensor([[[img.shape[2],img.shape[1]]]])).view(-1,4)
        rateos = bb[:,2] / (bb[:,3]+1e-8)
        areas = bb[:,2] * (bb[:,3]+1e-8)
        std, search = 0.2, True
        crop = None
        while search:
            good_box = (rateos-rateos.mean())**2  < rateos.std()**2 *std
            good_box &= (areas-(areas.mean()+areas.std()*(.4+std)/2))**2 < areas.std()**2 *(std/2)
            std += 0.1
            if good_box.any():
                bbs = bb[good_box].view(-1,4)
                for box in bbs:
                    box = torch.cat((box[:2]-box[2:]/2, box[:2]+box[2:]/2)).int()               # x1y1x2y2
                    crop = img[:, box[1]:box[3], box[0]:box[2]]
                    if crop.numel()>0:
                        search = False
                        break
            if std>1:break
        
        if crop is None or crop.numel()==0:
            return self._old_get_exemplar(img,target)
        return [crop]

    def _old_get_exemplar(self,img,target, p=0):
        bbnorm = target['boxes'][p].clone()
        bbnorm = bbnorm.clamp(min=0)
        bb = (bbnorm.view(2,2) * torch.tensor([img.shape[2],img.shape[1]]).view(1,2)).flatten()  # coords in img
        bb = torch.cat((bb[:2]-bb[2:]/2, bb[:2]+bb[2:]/2)).int()               # x1y1x2y2
        crop = img[:, bb[1]:bb[3], bb[0]:bb[2]]

        # check goodness of patch
        min_dim = torch.tensor([min(*crop.shape)],dtype=float)
        if len(target['boxes'])==p+1:
            # emergence
            crop = img[:, bb[1]:bb[1]+4, bb[0]:bb[0]+4]
        elif min_dim==0 or max(crop.shape[1:])/min(crop.shape[1:])>5:
            # get next box in case of errors
            return self._old_get_exemplar(img, target, p+1)
        return [crop]

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances



def build(image_set, args):
    root = pathlib.Path(args.fscd_path)
    assert root.exists(), f'provided FSCD path {root} does not exist'
    transform = make_synth_transforms(args, image_set)
    
    dataset = FSCDataset(args, image_set, transform)

    return dataset

