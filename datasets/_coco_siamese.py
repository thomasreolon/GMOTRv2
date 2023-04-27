
# Dataset class for COCO dataset
# Path: datasets/_coco_siamese.py
# Compare this snippet from engine/siamese2.py:
import torch
import os
from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np
import cv2

class SiameseDatset(torch.utils.data.Dataset):
    def __init__(self, coco_path, train=True, use_categories=True):
        self.split = 'train' if train else 'val'
        self.coco_path = coco_path
        self.train = train
        # self.use_categories = use_categories

        # initialize COCO api for instance annotations
        self.coco = COCO(coco_path+f'/annotations/instances_{self.split}2017.json')

        # get all categories
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.images_per_category = []
        for c in self.cats:
            img_ids = self.coco.getImgIds(catIds=[c['id']])
            if len(img_ids) < 3: continue
            self.images_per_category.append((img_ids, c['id']))    

    def __len__(self):
        return max(len(self.images_per_category), 150) if self.train else 10

    def __getitem__(self, index, fail=False, seed=55):
        """returns a list of positive and negative images"""
        index = index % len(self.images_per_category)
        N_POS_PATCH = 5
        if not self.train:
            index = 11-index
            state = np.random.get_state()
            np.random.seed(seed)
        try:
        
            # positive patches
            img_ids, c_id = self.images_per_category[index]
            img_id1, img_id2, img_id3 = np.random.choice(img_ids, 3, replace=False)
            patches = self.get_patches(img_id1, c_id) + self.get_patches(img_id2, c_id) + self.get_patches(img_id3, c_id)
            patches = self.clean_patches(patches, n_patch=N_POS_PATCH)

            # negative patches
            n_class = len(self.images_per_category)
            nindex = index +1 + np.random.randint(max(1, n_class-5))
            img_ids, c_id = self.images_per_category[nindex % n_class]
            img_idn = np.random.choice(img_ids)
            patchesn = self.get_patches(img_idn, c_id)
            img_ids, c_id = self.images_per_category[(nindex+3) % n_class]
            img_idn = np.random.choice(img_ids)
            patchesn += self.get_patches(img_idn, c_id)
            patchesn = self.clean_patches(patchesn, n_patch=3, hw=patches[0].shape[:2])

            # concatenate patches
            patches = patches + patchesn
            patches = np.stack(patches, axis=0)
            patches = torch.from_numpy(patches).float().permute(0,3,1,2) / 255  # (b,3,h,w)

            # normalize patches
            mean, var = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            patches = (patches - torch.tensor(mean).view(1,3,1,1)) / torch.tensor(var).view(1,3,1,1)
        except (TypeError, cv2.error) as e:
            if fail: raise e
            if not self.train:
                np.random.set_state(state)
            return self.__getitem__(index+500, True, seed+1)

        if not self.train:
            np.random.set_state(state)
        
        return patches, N_POS_PATCH

    def clean_patches(self, patches, n_patch=5, hw=None):
        """cleans the patches"""
        # remove patches with area < 32
        patches = [p for p in patches if p.shape[0] * p.shape[1] > 32]

        # remove patches too different in size
        if len(patches) > n_patch:
            avg_size = min(256, np.mean([(p.shape[0] * p.shape[1])**0.5 for p in patches])) if hw is None else (hw[0]*hw[1])**0.5
            patches = sorted(patches, key=lambda p: abs((p.shape[0] * p.shape[1])**0.5 - avg_size))
            patches = patches[:n_patch]
        
        # add patches to reach 5
        while len(patches) < n_patch:
            tmp = patches[0].copy()[:,::-1] + np.random.randn(*patches[0].shape)*10
            patches.append(tmp)
        
        # resize patches
        h,w = patches[0].shape[:2] if hw is None else hw
        h = max(33, h) ; w = max(33, w)
        patches = [letterbox(p,h,w) for p in patches]

        return patches

    def get_patches(self, img_id, cat_id):
        """extracts patches from the image"""
        # load image
        img_info = self.coco.loadImgs([img_id])[0]
        img = cv2.imread(os.path.join(self.coco_path, f'{self.split}2017', img_info['file_name']))

        # load annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        # extract patches
        patches = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            patches.append(img[y:y+h, x:x+w].astype(np.float32))
        return patches

    @staticmethod
    def collate_fn(batch):
        return batch[0]


def letterbox(img, h, w):
    """resize image with unchanged aspect ratio using padding"""
    shape = img.shape[:2]  # current shape [height, width]
    r = min(h / shape[0], w / shape[1])  # ratio  = new / old

    new_shape = (int(shape[1] * r), int(shape[0] * r))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)  # INTER_AREA is better, INTER_LINEAR is faster

    # create padding
    dw = w - new_shape[0]  # width padding
    dh = h - new_shape[1]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])