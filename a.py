from datasets import build_dataset
from configs.defaults import get_args_parser
import cv2
import numpy as np
import torch

args = get_args_parser().parse_args()
args.dataset_file = 'joint_coco_fscd_synth'
args.fscd_path = '../datasets/FSCD147'
args.synth_path = '../datasets/synth_data'
args.coco_path = '../datasets/coco'
args.sampler_lengths = [5]

ds = build_dataset('train', args)
print('ok', len(ds))

for i in range(100):
    i = (i*55339) % len(ds)
    d = ds[i]
    for img, gt in zip(d['imgs'], d['gt_instances']):
        img = (img-img.min()) / (img.max()-img.min())
        img = img.permute(1,2,0).numpy()[:,:,::-1]
        H,W,_ = img.shape
        def clean(x,X): return int(max(0,min(x, X-1)))

        for box, idx in zip(gt.boxes, gt.obj_ids):
            box = (box.view(2,2) * torch.tensor([W, H], device=box.device).view(1,2)).int()
            x1,x2 = box[0,0] - box[1,0].div(2,rounding_mode='trunc'), box[0,0] + box[1,0].div(2,rounding_mode='trunc')
            y1,y2 = box[0,1] - box[1,1].div(2,rounding_mode='trunc'), box[0,1] + box[1,1].div(2,rounding_mode='trunc')
            x1,x2,y1,y2 = clean(x1,W),clean(x2,W),clean(y1,H),clean(y2,H)
            tmp = img[y1+1:y2-1, x1+1:x2-1].copy()
            img[y1:y2+1, x1:x2+1] = tuple([(((5+idx.item()*3)*4909 % p)%256) for p in (3001, 1109, 2027)])
            img[y1+1:y2-1, x1+1:x2-1] = tmp
            

        cv2.imshow('f', np.uint8(img*255))
        cv2.waitKey()

print('end')