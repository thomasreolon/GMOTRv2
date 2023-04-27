
# Dataset class for COCO dataset
# Path: datasets/_coco_siamese.py
# Compare this snippet from engine/siamese2.py:
import torch
import os
from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np
import cv2
from PIL import Image
import torchvision
from collections import defaultdict

class DetectionDatset(torch.utils.data.Dataset):
    def __init__(self, coco_path, train=True):
        self.split = 'train' if train else 'val'
        self.coco_path = coco_path
        self.train = train
        self.transform =   torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])  if train else torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])


        # initialize COCO api for instance annotations
        self.coco = COCO(coco_path+f'/annotations/instances_{self.split}2017.json')

        # get all categories
        shapes = defaultdict(list)
        batches = {}
        bs_max = 427*640
        for idx in self.coco.getImgIds():
            info = self.coco.loadImgs(idx)[0]
            boxes = [x['bbox'] for x in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[idx]))]
            if all([max(box[2:])<33 for box in boxes]): continue
            sh = f'{info["width"]}-{info["height"]}'
            pixels = info["width"]*info["height"]
            if pixels>bs_max*1.8: continue
            bs = 2 if pixels>bs_max else 4
            shapes[sh].append(idx)
            batches[sh] = bs
        self.indexes = [v for v in shapes.values() if len(v) > 15]
        self.bs = [v for v in batches.values()]
        n=sum([len(v) for v in self.indexes])
        self.pr = [len(v)/n for v in self.indexes]


    def __len__(self):
        return 500 if self.train else 10

    def get_image(self, idx):
        img_info = self.coco.loadImgs([idx])[0]
        img = Image.open(os.path.join(self.coco_path, self.split+'2017', img_info['file_name']))
        img = img.convert('RGB')
        img = self.transform(img)
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[idx]))
        annotations = [a for a in annotations if min(*a['bbox'][2:])>33]
        if len(annotations)==0: raise IdxException('no annotations')
        cat_id = [a['category_id'] for a in annotations][np.random.randint(len(annotations))]
        annotations = [a for a in annotations if a['category_id'] == cat_id]
        boxes = torch.tensor([a['bbox'] for a in annotations]).float().view(-1, 4)
        return img, boxes


    def __getitem__(self, index, fail=10, seed=55):
        """returns a list of positive and negative images"""
        if not self.train:
            state = np.random.get_state()
            np.random.seed(seed+index)

        try:
            # get images with same shape
            i = np.random.choice(list(range(len(self.indexes))), p=self.pr)
            indexes = self.indexes[i]
            bs = self.bs[i]
            # get 4 images
            indexes = np.random.choice(indexes, bs, replace=False)

            # get images
            res = [self.process(i) for i in indexes]
            data_dict =  self.concat(res)
            if not self.train: np.random.set_state(state)
            return data_dict

        except Exception as e:
            if fail<0: raise e
            if not self.train: np.random.set_state(state)
            return self.__getitem__(index+1, fail=fail-1, seed=seed+1)

    def concat(self, raw_res):
        res = {'imgs': [], 'boxes': [], 'exemplars': []}
        shapes = np.array([0,0,0])
        for i, (img, boxes, exemplars) in enumerate(raw_res):
            res['imgs'].append(img)
            boxes = torch.cat([torch.ones_like(boxes[:, :1])*i, boxes], dim=1)
            res['boxes'].append(boxes)
            res['exemplars'].append(exemplars)
            for ex in exemplars:
                shapes[0] += ex.shape[1]
                shapes[1] += ex.shape[2]
                shapes[2] += 1
        avg_shape = shapes[:2] / shapes[2]
        res['exemplars'] = [sorted(exs, key=lambda x:(x.shape[1]-avg_shape[0])**2+(x.shape[2]-avg_shape[1])**2)[:2] for exs in res['exemplars']]
        res['exemplars'] = [[letterbox(ex, int(avg_shape[1]), int(avg_shape[0])) for ex in exs] for exs in res['exemplars']]

        self.copypaste(res['exemplars'], res['imgs'], res['boxes'])

        res['exemplars'] = [torch.stack(a) for a in res['exemplars']]
        res['exemplars'] = torch.stack(res['exemplars'])
        res['boxes'] = torch.cat(res['boxes'], dim=0)
        res['imgs'] = torch.stack(res['imgs'], dim=0)
        return res

    def copypaste(self, exemplars, imgs, lboxes):
        # paste exemplar on image
        for i, (img, boxes) in enumerate(zip(imgs, lboxes)):
            if len(boxes[boxes[:,0]==i])>1:continue
            if exemplars[i][0].shape[1]>img.shape[1]: continue
            if exemplars[i][0].shape[2]>img.shape[2]: continue
            patch = torchvision.transforms.functional.hflip(exemplars[i][0])  # hflip
            box_position = boxes[0, 1:3].numpy()
            d1 = ((box_position-(0,0.5))**2).sum()
            d2 = ((box_position-(1,0))**2).sum()
            d3 = ((box_position-(1,1))**2).sum()
            r = np.random.rand()
            if d1>=d2 and d1>=d3:
                p = box_position*r
                p[1] += (1-r)*0.5
            elif d2>d1 and d2>d3:
                p = box_position*r
                p[0] += (1-r)
            else:
                p = box_position*r + (1-r)

            # draw patch
            p = (p*(img.shape[2], img.shape[1])).astype(int)
            p[0] = min(max(p[0], 0), img.shape[2]-patch.shape[2])
            p[1] = min(max(p[1], 0), img.shape[1]-patch.shape[1])
            img[:, p[1]:p[1]+patch.shape[1], p[0]:p[0]+patch.shape[2]] = patch

            # add box
            new_box = torch.tensor([[i, p[0]+patch.shape[2]/2, p[1]+patch.shape[1]/2, patch.shape[2], patch.shape[1]]]).float()
            new_box = new_box / torch.tensor([1, img.shape[2], img.shape[1], img.shape[2], img.shape[1]]).float()

            lboxes[i] = torch.cat([lboxes[i], new_box], dim=0)




    def process(self, idx):
        img, boxes = self.get_image(idx)
        boxes[:,2:] += boxes[:, :2]

        # get exemplars
        exemplars = []
        for box in boxes:
            patch = img[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if min(patch.shape[1:]) >32:
                exemplars.append(patch)
        if len(exemplars)==0:
            raise IdxException('no exemplars')
        elif len(exemplars)==1:
            exemplars.append(exemplars[0])

        # from top-left to center
        boxes[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        boxes[:, 2:] = (boxes[:, 2:]-boxes[:, :2])*2

        # normalize boxes
        boxes = boxes / torch.tensor([img.shape[2], img.shape[1], img.shape[2], img.shape[1]]).float().view(1, -1)

        return img, boxes, exemplars




    @staticmethod
    def collate_fn(batch):
        return batch[0]

def letterbox(img, h, w):
    """resize image with unchanged aspect ratio using padding"""
    img = img.permute(1, 2, 0).numpy()
    shape = img.shape[:2]  # current shape [height, width]
    r = min(h / shape[0], w / shape[1])  # ratio  = new / old

    new_shape = (int(shape[1] * r), int(shape[0] * r))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)  # INTER_AREA is better, INTER_LINEAR is faster

    # create padding
    dw = w - new_shape[0]  # width padding
    dh = h - new_shape[1]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


class IdxException(Exception):
    pass