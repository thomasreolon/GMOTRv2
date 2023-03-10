from collections import defaultdict
from PIL import Image
import pathlib
import json

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from .transforms import make_imgdataset_transforms
from models.structures import Instances


class FSCDataset(Dataset):
    selected = ['23.jpg', '60.jpg', '76.jpg', '97.jpg', '98.jpg', '367.jpg', '417.jpg', '487.jpg', '783.jpg', '791.jpg', '881.jpg', '2103.jpg', '2130.jpg', '2415.jpg', '2423.jpg', '2435.jpg', '2441.jpg', '2458.jpg', '2461.jpg', '2477.jpg', '2478.jpg', '2482.jpg', '2491.jpg', '2499.jpg', '2517.jpg', '2520.jpg', '2529.jpg', '2538.jpg', '2558.jpg', '2664.jpg', '2698.jpg', '2788.jpg', '2793.jpg', '2800.jpg','3084.jpg', '3085.jpg', '3091.jpg', '3149.jpg', '3184.jpg', '3193.jpg', '3194.jpg', '3196.jpg', '3197.jpg', '3198.jpg', '3207.jpg', '3215.jpg', '3255.jpg', '3294.jpg', '3298.jpg', '3303.jpg', '3304.jpg', '3305.jpg', '3364.jpg', '3368.jpg', '3371.jpg', '3372.jpg', '3384.jpg', '3387.jpg', '3408.jpg', '3445.jpg', '3607.jpg', '3633.jpg', '3681.jpg', '3684.jpg', '3692.jpg', '3703.jpg', '3708.jpg', '3721.jpg', '3734.jpg', '3740.jpg', '3798.jpg', '3814.jpg', '3826.jpg', '3828.jpg', '3829.jpg', '3882.jpg', '3897.jpg', '3949.jpg', '3952.jpg', '3960.jpg', '3964.jpg', '3973.jpg', '4127.jpg', '4136.jpg', '4182.jpg', '4187.jpg', '4190.jpg', '4203.jpg', '4207.jpg', '4209.jpg', '4210.jpg', '4212.jpg', '4217.jpg', '4232.jpg', '4238.jpg', '4243.jpg', '4247.jpg', '4252.jpg', '4253.jpg', '4254.jpg', '4256.jpg', '4258.jpg', '4260.jpg', '4262.jpg', '4269.jpg', '4271.jpg', '4274.jpg', '4275.jpg', '4276.jpg', '4278.jpg', '4306.jpg', '4333.jpg', '4337.jpg', '4353.jpg', '4372.jpg', '4430.jpg', '4431.jpg', '4436.jpg', '4479.jpg', '4480.jpg', '4621.jpg', '4628.jpg', '4666.jpg', '4782.jpg', '4864.jpg', '4973.jpg', '4978.jpg', '4980.jpg', '4981.jpg', '4984.jpg', '5030.jpg', '5033.jpg', '5041.jpg', '5125.jpg', '5130.jpg', '5138.jpg', '5159.jpg', '5162.jpg', '5182.jpg', '5243.jpg', '5244.jpg', '5245.jpg', '5248.jpg', '5253.jpg', '5271.jpg', '5307.jpg', '5348.jpg', '5354.jpg', '5412.jpg', '5421.jpg', '5423.jpg', '5425.jpg', '5427.jpg', '5430.jpg', '5447.jpg', '5453.jpg', '5456.jpg', '5457.jpg', '5536.jpg', '5603.jpg', '5619.jpg', '5625.jpg', '5632.jpg', '5709.jpg', '5713.jpg', '5717.jpg', '5732.jpg', '5751.jpg', '5779.jpg', '5780.jpg', '5959.jpg', '5961.jpg', '5963.jpg', '5967.jpg', '5984.jpg', '5988.jpg', '5994.jpg', '6008.jpg', '6022.jpg', '6028.jpg', '6030.jpg', '6177.jpg', '6198.jpg', '6205.jpg', '6212.jpg', '6222.jpg', '6230.jpg', '6375.jpg', '6378.jpg', '6387.jpg', '6390.jpg', '6391.jpg', '6407.jpg', '6420.jpg', '6425.jpg', '6438.jpg', '6445.jpg', '6448.jpg', '6456.jpg', '6473.jpg', '6493.jpg', '6495.jpg', '6504.jpg', '6529.jpg', '6558.jpg', '6587.jpg', '6683.jpg', '6687.jpg', '7177.jpg', '6972.jpg', '7638.jpg', '7158.jpg', '6975.jpg', '7170.jpg', '7227.jpg', '7166.jpg', '7141.jpg', '237.jpg', '591.jpg', '984.jpg',  '1910.jpg', '1972.jpg', '1982.jpg', '2815.jpg', '2825.jpg', '3266.jpg', '3427.jpg', '3432.jpg', '3438.jpg', '3478.jpg', '3518.jpg', '3546.jpg', '3567.jpg', '3660.jpg', '3661.jpg', '3669.jpg', '3675.jpg', '3762.jpg', '3764.jpg', '3768.jpg', '3770.jpg', '3771.jpg', '3780.jpg', '3782.jpg', '4585.jpg', '4754.jpg', '5048.jpg', '5059.jpg', '5062.jpg', '5096.jpg', '5152.jpg', '5155.jpg', '5202.jpg', '5210.jpg', '5223.jpg', '5664.jpg', '5668.jpg', '5670.jpg', '5900.jpg', '6101.jpg', '6107.jpg', '6120.jpg', '7403.jpg', '7487.jpg', '7504.jpg', '7049.jpg', '3.jpg', '311.jpg', '996.jpg', '2032.jpg', '2143.jpg', '2181.jpg', '2213.jpg', '2227.jpg', '2269.jpg', '2288.jpg', '2908.jpg', '2912.jpg', '2918.jpg', '2924.jpg', '2942.jpg', '2945.jpg', '4063.jpg', '4073.jpg', '4086.jpg', '4120.jpg', '4287.jpg', '4294.jpg', '4303.jpg', '4935.jpg', '5380.jpg', '5387.jpg', '5463.jpg', '5464.jpg', '5512.jpg', '5575.jpg', '5917.jpg', '5923.jpg', '5927.jpg', '6042.jpg', '7693.jpg', '7273.jpg', '7263.jpg', '7384.jpg', '7709.jpg', '7080.jpg', '7683.jpg', '7581.jpg', '7666.jpg', '7282.jpg', '6856.jpg', '7506.jpg']
    def __init__(self, args, split, transform) -> None:
        super().__init__()
        self.args = args
        self._cache = {}
        self.transform = transform

        # folder with images
        self.path_imgs = args.fscd_path+'/images_384_VarV2/'

        # get data from json
        det2 = json.load(open(args.fscd_path+'/instances_test.json', 'r'))      # used for train
        det = json.load(open(args.fscd_path+'/instances_val.json', 'r'))        # used for test/val
        all_ann = json.load(open(args.fscd_path+'/annotation_FSC147_384.json', 'r'))

        # select train/test  (a bit ugly cause we have GT only for some images)
        if args.small_dataset:
            for k,v in det2.items():
                if k=='images': v=[{**img, 'id':img['id']+100000} for img in v]
                if k=='annotations': v=[{**img, 'image_id':img['image_id']+100000} for img in v]
                det[k] += v
            selected = set(self.selected[:-10] if split=='train' else self.selected[-10:])

        # load listof(images,detections)
        self.detections = self.load_anns(all_ann, det) + self.load_anns(all_ann, det2)

        if args.small_dataset:
            self.detections = [d for d in self.detections  if d[0] in selected]

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
        if idx in self._cache:
            img, target, exe = self._cache[idx]
        else:
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

            if len(self._cache) < 3000:
                self._cache[idx] = img, target, exe


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
    transform = make_imgdataset_transforms(args, image_set)
    
    dataset = FSCDataset(args, image_set, transform)

    return dataset

