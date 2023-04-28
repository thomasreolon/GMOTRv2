"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import numpy as np

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

# from util.misc import get_local_rank, get_local_size
import datasets._transforms as T

from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
from models.structures import Instances

class TvCocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(TvCocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = False
        self.local_rank = local_rank
        self.local_size = local_size

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, args, transforms, return_masks, full_dataset=False, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self._full_dataset = full_dataset
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.set_epoch(0)

    def __len__(self):
        return len(self.act_ids)

    def set_epoch(self, epoch):
        ssize = 1 if self._full_dataset else 10
        s, b = len(self.ids)//ssize, epoch%ssize
        self.act_ids = list(range(b*s, (b+1)*s))
        self.current_epoch = epoch

    def step_epoch(self):
        # one epoch finishes.
        self.set_epoch(self.current_epoch + 1)
    @staticmethod
    def collate_fn(batch):
        """Forces collate a batch 1 element at a time"""
        assert len(batch) == 1, "Batch size must be 1"
        return batch[0]


    def __getitem__(self, idx, allow_fails=3):
        idx = self.act_ids[idx % len(self.act_ids)]
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if img is None: # image has no valid BB
            return self.__getitem__(idx-1, allow_fails=allow_fails)
        if self._transforms is not None:
            images, targets = self._transforms([img], [target])

        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)

        if any([ t['boxes'].shape[0]==0 for t in targets ]):
            if allow_fails==0: idx = idx+53
            return self.__getitem__(idx, True)

        exemplar = self.get_exemplar(images[0], targets[0])
        if len(exemplar)==0:
            if allow_fails: return self.__getitem__(idx-2, allow_fails=allow_fails-1)
            else:          raise Exception('No exemplar found')

        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'exemplar': exemplar,
        }

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

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


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # keep good BB  or one class
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if len(classes[keep])==0: return None, None #### no valid BB
        sel_clas = np.random.choice(classes[keep])
        keep = keep & (classes == sel_clas)

        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.zeros_like(classes)
        target["obj_ids"] = torch.arange(len(target["boxes"])).long()
        if self.return_masks:
            target["masks"] = masks

        return image, target



def build(image_set, args, full_dataset=False):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, args, transforms=T.make_coco_transforms(args, image_set), 
                            return_masks=False, full_dataset=full_dataset,  cache_mode=False)
    return dataset
