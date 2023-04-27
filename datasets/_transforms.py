# ------------------------------------------------------------------------
# Copyright (c) 2022 RIKEN. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Transforms and data augmentation for both image + bbox.
"""
import copy
import random
import PIL
import numpy as np
import cv2

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.ops.misc import interpolate

from util.box_ops import box_xyxy_to_cxcywh

def make_synth_transforms(args, image_set):
    scales = [608]#, 800, 1080]
    if image_set == 'train':
        return MotCompose([
            MotRandomHorizontalFlip(),
            MotRandomResize(scales, max_size=1920),
            MOTHSV(),
            MotToTensor(),
            MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return MotCompose([
            MotToTensor(),
            MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def make_coco_transforms(args, image_set):
    scales = [540, 608]#, 726, 1021, 962, 1080]
    if image_set == 'train':
        return MotCompose([
            MotRandomHorizontalFlip(),
            MotRandomResize(scales, max_size=1920),
            MotCopyPaste(),
            MotRandomShift(args.sampler_lengths),
            MOTHSV(),
            MotToTensor(),
            MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            MOTCleanGT(),
        ])
    else:
        return MotCompose([
            MotRandomShift(args.sampler_lengths, test=True),
            MotToTensor(),
            MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            MOTCleanGT(),
        ])


# def make_imgdataset_transforms(args, image_set, use_moving_crop=True):
#     normalize = MotCompose([
#         MotToTensor(),
#         MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     scales = [608, 640, 672, 704, 736, 768, 800, 832, 864]

#     if image_set == 'train':
#         if use_moving_crop:
#             return MotCompose([
#                 MotRandomHorizontalFlip(),
#                 MotRandomResize(scales, max_size=1555),
#                 MotRandomShiftExtender(1,args.sampler_lengths),
#                 MotMovingRandomCrop(),
#                 MOTHSV(),
#                 normalize,  # also scales from HW to [01]
#                 MOTCleanGT(),
#             ])
#         else:
#             return MotCompose([
#                 MotRandomHorizontalFlip(),
#                 MotRandomResize(scales, max_size=1555),
#                 MotRandomShiftExtender(1,args.sampler_lengths),
#                 MOTHSV(),
#                 normalize,  # also scales from HW to [01]
#                 MOTCleanGT(),
#             ])

#     else:
#         return MotCompose([
#             MotRandomShiftExtender(1,args.sampler_lengths),
#             MotRandomResize([800], max_size=1333),
#             normalize,
#         ])

class MotCopyPaste():
    def __call__(self, imgs: list, targets: list):
        base_img, base_target = imgs[0], targets[0]
        if len(base_target['boxes']) == 0:
            return imgs, targets
        box = base_target['boxes'][0]
        patch = base_img.crop(box.tolist())
        r = np.random.rand() + 0.2
        n_w = (box[2]-box[0])*r ; n_h = (box[3]-box[1])*r
        if n_w > base_img.size[0]*.5 or n_h > base_img.size[1]*.5:
            n_w /= 2 ; n_h /= 2
        patch = patch.resize((int(n_w), int(n_h)))
        patch = patch.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        x,y = self.get_coords(base_img, patch, base_target)
        base_img.paste(patch, (x,y))
        base_target['boxes'] = torch.cat([base_target['boxes'], torch.tensor([x,y,x+patch.size[0],y+patch.size[1]]).unsqueeze(0)], dim=0)
        base_target['labels'] = torch.cat([base_target['labels'], torch.tensor([base_target['labels'][0]])], dim=0)
        base_target['obj_ids'] = torch.cat([base_target['obj_ids'], torch.tensor([max(*base_target['obj_ids'].tolist(),0)+1])], dim=0)
        return imgs, targets

    def get_coords(self, img, patch, target):
        x,y = np.random.randint(0, 1+img.size[0]-patch.size[0]), np.random.randint(0, 1+img.size[1]-patch.size[1])
        boxes = target['boxes']

        # Check if the patch ovverlaps with any bounding box
        min_out = 4
        while ((boxes[:,0]+min_out > x) & (boxes[:,1]+min_out> y) & (boxes[:,2]-min_out < x+patch.size[0]) & (boxes[:,3]-min_out < y+patch.size[1])).any():
            if np.random.rand() > 0.5:
                patch = patch.resize((int(patch.size[0]*0.8), int(patch.size[1]*0.8)))
            x,y = np.random.randint(0, 1+img.size[0]-patch.size[0]), np.random.randint(0, 1+img.size[1]-patch.size[1])
        return x,y


class MotRandomShift():
    def __init__(self, n_imgs, test: bool= False):
        self.n_imgs = n_imgs
        self.test = test

    def is_large_bounding_box(self, img, target):
        box = target['boxes'][0]
        area = (box[2]-box[0])*(box[3]-box[1])
        return area > 0.5*img.size[0]*img.size[1]

    def __call__(self, imgs: list, targets: list):
        if self.is_large_bounding_box(imgs[0], targets[0]):
            # Skip if the first image has a large bounding box
            return imgs, targets
        
        ret_imgs = []
        ret_targets = []
        w, h = imgs[0].size
        r = np.random.rand()
        w, h = int((0.9+r*0.1)*w), int((0.9+r*0.1)*h)

        # deterministic or random
        r = 0.9 if self.test else np.random.rand()
        r = r*.2+.8     # 80/90 % of the image
        
        # deterministic or random
        if self.test:
            i,j = imgs[0].size[1]-int(r*h), imgs[0].size[0]-int(r*w)
            region1 = i//2, j//2, int(r*h),int(r*w)
            region2 = i//3, j//3, int(r*h),int(r*w)
        else: 
            region1 = T.RandomCrop.get_params(imgs[0], [int(r*h),int(r*w)])
            region2 = T.RandomCrop.get_params(imgs[0], [int(r*h),int(r*w)])

        base_img, base_target = imgs[0], targets[0]
        for i in range(self.n_imgs):
            img_i, target_i = copy.deepcopy(base_img), copy.deepcopy(base_target)
            x = region2[0]*i/(self.n_imgs) + region1[0]*(self.n_imgs-i)/(self.n_imgs)
            y = region2[1]*i/(self.n_imgs) + region1[1]*(self.n_imgs-i)/(self.n_imgs)
            region = int(x), int(y), int(r*h), int(r*w)
            img_i, target_i = random_shift(base_img, base_target, region, (h,w))
            ret_imgs.append(img_i)
            ret_targets.append(target_i)

        return ret_imgs, ret_targets

def random_shift(image, target, region, sizes):
    oh, ow = sizes
    # step 1, shift crop and re-scale image firstly
    cropped_image = F.crop(image, *region)
    cropped_image = F.resize(cropped_image, sizes, interpolation=F.InterpolationMode.NEAREST)

    target = target.copy()
    i, j, h, w = region

    # update boxes
    cropped_boxes = target["boxes"] - torch.as_tensor([j, i, j, i])
    cropped_boxes *= torch.as_tensor([ow / w, oh / h, ow / w, oh / h])
    cropped_boxes = cropped_boxes.reshape(-1, 2, 2)
    max_size = torch.as_tensor([ow-1, oh-1], dtype=torch.float32)
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    target["boxes"] = cropped_boxes.view(-1,4)

    # select by cut
    hwratio = torch.tensor(list(map(lambda box:  (box[3]-box[1])/(box[2]-box[0]+1e-4), cropped_boxes.view(-1, 4))))
    not_mostly_cut = (hwratio < 6) & (hwratio > 1/6)

    # select by area #NOTE: kill bounding boxes less 20 pixels
    areas = torch.tensor(list(map(lambda box:  (box[3]-box[1])*(box[2]-box[0]), cropped_boxes.view(-1, 4))))
    big_enough = areas >= 20

    keep = big_enough & not_mostly_cut

    always_keep = {'size', 'ori_img', 'image_id', 'orig_size', 'area'}
    for field in target.keys():
        if field not in always_keep:
            target[field] = target[field][keep]

    return cropped_image, target









def make_viddataset_transforms(args, image_set):

    normalize = MotCompose([
        MotToTensor(),
        MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return MotCompose([
            MotRandomHorizontalFlip(),
            MotRandomSelect(
                MotRandomResize(scales, max_size=1536),
                MotCompose([
                    MotRandomResize([800, 1000, 1200]),
                    FixedMotRandomCrop(800, 1200),
                    MotRandomResize(scales, max_size=1536),
                ])
            ),
            MOTHSV(),
            normalize,
        ])

    else:
        return MotCompose([
            MotRandomResize([800], max_size=1333),
            normalize,
        ])

class MotMovingRandomCrop():
    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []
        image_width, image_height = imgs[0].size
        c1,c2,r = [(3,4,1),(1,2,0.5),(5,6,1),(1,2,0.2)] [int(np.random.rand(1)*4)]

        w,h = image_width*c1//c2, image_height*c1//c2
        j = int(np.random.rand(1)*image_width/c2)
        i = int(np.random.rand(1)*image_height/c2)
        step_j = (-j if image_width-j-w < j else image_width-j-w)/(len(imgs)-0.999)*r
        step_i = (-i if image_height-i-h < i else image_height-i-h)/(len(imgs)-0.999)*r

        for img_i, targets_i in zip(imgs, targets):
            region = i,j,h,w
            img_i, targets_i = crop(img_i, targets_i, region)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
            i += int(np.random.rand(1)*step_i)
            j += int(np.random.rand(1)*step_j)
        return ret_imgs, ret_targets


def crop_mot(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    fields = ["labels", "obj_ids"]

    boxes = target["boxes"]
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    target["boxes"] = cropped_boxes.reshape(-1, 4)
    fields.append("boxes")

    cropped_boxes = target['boxes'].reshape(-1, 2, 2)
    keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

    for field in fields:
        n_size = len(target[field])
        target[field] = target[field][keep[:n_size]]

    return cropped_image, target



def random_shift_noresize(image, target, region):
    ow, oh = image.size
    # step 1, shift crop and re-scale image firstly
    shifted_image = F.crop(image, *region)

    padding = [
        ow - region[1]-region[3],  # left
        oh - region[0]-region[2], # top
        region[1], # right
        region[0], # bott
    ]
    shifted_image = F.pad(shifted_image, padding)

    target = target.copy()

    # translations due to padding
    j, i = padding[0]-padding[2], padding[1]-padding[3]

    fields = ["labels", "iscrowd", "obj_ids"]

    if "boxes" in target:
        boxes = target["boxes"]
        shifted_boxes = boxes + torch.as_tensor([j, i, j, i])
        # shifted_boxes *= torch.as_tensor([ow / w, oh / h, ow / w, oh / h])
        target["boxes"] = shifted_boxes.reshape(-1, 4)
        fields.append("boxes")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            shifted_boxes = target['boxes'].reshape(-1, 2, 2)
            max_size = torch.as_tensor([ow, oh], dtype=torch.float32)
            shifted_boxes = torch.min(shifted_boxes.reshape(-1, 2, 2), max_size)
            shifted_boxes = shifted_boxes.clamp(min=0)
            keep = torch.all(shifted_boxes[:, 1, :] > shifted_boxes[:, 0, :]+4, dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            n_size = len(target[field])
            target[field] = target[field][keep[:n_size]]

    return shifted_image, target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region


    fields = ["labels", "obj_ids"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)

        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            areas = cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]
            areas = (areas[:, 0] * areas[:, 1])
            small = areas < areas.mean()*0.5
            touch_edge_l = (cropped_boxes[:, 0, 0]==0) | (cropped_boxes[:, 0, 1]==0)
            touch_edge_r = (cropped_boxes[:, 1, 0]==max_size[0]) | (cropped_boxes[:, 1, 1]==max_size[1])
            keep = keep & ~(small & (touch_edge_l | touch_edge_r))
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    boxes = target["boxes"]
    scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    target["boxes"] = scaled_boxes

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()

    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class MOTHSV:
    def __init__(self, hgain=5, sgain=30, vgain=30) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, imgs: list, targets: list):
        hsv_augs = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # random gains
        hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
        hsv_augs = hsv_augs.astype(np.int16)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)

            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

            imgs[i] = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2RGB)  # no return needed
        return imgs, targets


class MOTCleanGT:
    def __init__(self) -> None:
        pass

    def __call__(self, imgs: list, targets: list):
        max_ = 1-1e-9

        for target in targets:

            bbs = target['boxes']
            tmp = bbs[:, :2].clamp(0,max_)
            diff = (bbs[:, :2] - tmp).abs()
            coeff = bbs[:, 2:] / (diff*2+bbs[:, 2:])

            bbs[:, :2] = tmp
            bbs[:, 2:] *= coeff
            
            keep = ((bbs>=1).sum(-1) + (bbs<0).sum(-1)) == 0

            always_keep = {'size', 'ori_img', 'image_id', 'orig_size', 'area'}
            for field in target.keys():
                if field not in always_keep:
                    target[field] = target[field][keep]

        return imgs, targets


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class MotRandomCrop(RandomCrop):
    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []
        region = T.RandomCrop.get_params(imgs[0], self.size)
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop(img_i, targets_i, region)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets

class FixedMotRandomCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []
        w = random.randint(self.min_size, min(imgs[0].width, self.max_size))
        h = random.randint(self.min_size, min(imgs[0].height, self.max_size))
        region = T.RandomCrop.get_params(imgs[0], [h, w])
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop_mot(img_i, targets_i, region)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class MotRandomSizeCrop(RandomSizeCrop):
    def __call__(self, imgs, targets):
        w = random.randint(self.min_size, min(imgs[0].width, self.max_size))
        h = random.randint(self.min_size, min(imgs[0].height, self.max_size))
        region = T.RandomCrop.get_params(imgs[0], [h, w])
        ret_imgs = []
        ret_targets = []
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop(img_i, targets_i, region)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class MotCenterCrop(CenterCrop):
    def __call__(self, imgs, targets):
        image_width, image_height = imgs[0].size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        ret_imgs = []
        ret_targets = []
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop(img_i, targets_i, (crop_top, crop_left, crop_height, crop_width))
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class MotRandomHorizontalFlip(RandomHorizontalFlip):
    def __call__(self, imgs, targets):
        if random.random() < self.p:
            ret_imgs = []
            ret_targets = []
            for img_i, targets_i in zip(imgs, targets):
                img_i, targets_i = hflip(img_i, targets_i)
                ret_imgs.append(img_i)
                ret_targets.append(targets_i)
            return ret_imgs, ret_targets
        return imgs, targets


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class MotRandomResize(RandomResize):
    def __call__(self, imgs, targets):

        # images too big cause CUDA OOM --> images with this number of pixels (730*1000) are still supported in a 8GB GPU
        size = random.choice(self.sizes)

        # once we get the size we resize each image
        ret_imgs = []
        ret_targets = []
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = resize(img_i, targets_i, size, self.max_size)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class MotRandomPad(RandomPad):
    def __call__(self, imgs, targets):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        ret_imgs = []
        ret_targets = []
        for img_i, targets_i in zip(imgs, targets):
            img_i, target_i = pad(img_i, targets_i, (pad_x, pad_y))
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class MotRandomSelect(RandomSelect):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __call__(self, imgs, targets):
        if random.random() < self.p:
            return self.transforms1(imgs, targets)
        return self.transforms2(imgs, targets)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class MotToTensor(ToTensor):
    def __call__(self, imgs, targets):
        ret_imgs = []
        for img in imgs:
            ret_imgs.append(F.to_tensor(img))
        return ret_imgs, targets


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class MotRandomErasing(RandomErasing):
    def __call__(self, imgs, targets):
        # TODO: Rewrite this part to ensure the data augmentation is same to each image.
        ret_imgs = []
        for img_i, targets_i in zip(imgs, targets):
            ret_imgs.append(self.eraser(img_i))
        return ret_imgs, targets


class MoTColorJitter(T.ColorJitter):
    def __call__(self, imgs, targets):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        ret_imgs = []
        for img_i, targets_i in zip(imgs, targets):
            ret_imgs.append(transform(img_i))
        return ret_imgs, targets


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if target is not None:
            target['ori_img'] = image.clone()
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class MotNormalize(Normalize):
    def __call__(self, imgs, targets=None):
        ret_imgs = []
        ret_targets = []
        for i in range(len(imgs)):
            img_i = imgs[i]
            targets_i = targets[i] if targets is not None else None
            img_i, targets_i = super().__call__(img_i, targets_i)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class MotCompose(Compose):
    def __call__(self, imgs, targets):
        for t in self.transforms:
            imgs, targets = t(imgs, targets)
        return imgs, targets
