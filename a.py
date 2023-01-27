from datasets import build_dataset
from configs.defaults import get_args_parser
import cv2
import numpy as np

args = get_args_parser().parse_args()
args.dataset_file = 'joint_synth'
args.fscd_path = '../dataset/FSCD147'
args.synth_path = '../dataset/synth_data'
args.sampler_lengths = [5]

ds = build_dataset('train', args)
print('ok', len(ds))

for i in range(100):
    d = ds[i]
    for img in d['imgs']:
        img = (img-img.min()) / (img.max()-img.min())
        img = img.permute(1,2,0).numpy()[:,:,::-1]
        cv2.imshow('f', np.uint8(img*255))
        cv2.waitKey()

print('end')