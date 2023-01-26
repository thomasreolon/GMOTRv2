# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os, numpy as np
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from models import build_model
from main import get_args_parser

from tools.visualize import process2 
from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from configs.defaults import get_args_parser
from datasets.fscd import build as build_fscd

from util.plot_utils import visualize_pred, visualize_gt, train_visualize_pred
from util.eval import compute_mota

def main():
    # Info about code execution
    args = get_args_parser().parse_args()
    if os.path.exists(f'{args.output_dir}/results/{args.resume.split("/")[-1][:-4]}.csv'):
        print("##"*50, f"\n#  Already Processed: {args.resume} #\n", "##"*50, sep='') ; exit(0)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build Model
    model = load_for_eval(args).to(args.device).eval()
    model.track_embed.score_thr = args.prob_detect*.8

    print('------ loading dataset  ------ ')
    dataset = load_svdataset('e2e_gmot', 'val', args)

    det = Detector(args, model, dataset)
    print('------  tracking  ------ ')
    for vid in range(len(dataset)):
        det.detect(vid)

    print('------  getting scores ------ ')
    compute_mota(args.output_dir, args.resume.split('/')[-1][:-4], args.gmot_path+'/track_label/', str(det.predict_path)+'/')

    print('------  making video  ------ ')
    process2(str(det.predict_path)+'/', args.gmot_path+'/GenericMOT_JPEG_Sequence/', args.output_dir+f'/video_{args.resume.split("/")[-1][:-4]}.mp4')

def load_svdataset(datasetname, split, args):
    assert datasetname in {'e2e_gmot', 'e2e_fscd'}, f'invalid dataset "{datasetname}"'
    assert split in {'train', 'val', 'test'}, f'invalid dataset "{split}"'
    
    if datasetname=='e2e_gmot':
        return load_gmot(split, args)
    elif datasetname=='e2e_fscd':
        return load_fscd(split, args)

def load_gmot(split, args):
    base_path = args.gmot_path+'/GenericMOT_JPEG_Sequence'

    list_dataset = []
    videos = os.listdir(base_path)

    for video in videos:
        # get 1st 12 BB
        gt = args.gmot_path+f'/track_label/{video}.txt'
        with open(gt, 'r') as fin:
            lines = []
            for _ in range(300):
                line = fin.readline()
                if line[0] == '0': 
                    lines.append([int(l) for l in line.split(',')])
                if len(lines)==12:break
        
        # select good BB
        idx = exemplar_n[video]
        line = lines[idx]

        # get coords
        bb = line[2], line[3], line[2]+line[4], line[3]+line[5],

        # get images
        imgs = sorted(os.listdir(f'{base_path}/{video}/img1'))

        ## visualize Exemplar
        # x=cv2.imread(f'{base_path}/{video}/img1/{imgs[0]}')
        # cv2.imshow('video', x)
        # # for i, line in enumerate(lines):
        # #     bb = line[2], line[3], line[2]+line[4], line[3]+line[5]
        # cv2.imshow('e', x[bb[1]:bb[3], bb[0]:bb[2]])
        # cv2.waitKey()

        list_dataset.append((f'{base_path}/{video}/img1/', imgs, bb))  # none should be the exemplar_bb xyxy

    return list_dataset


def load_fscd(split, args):
    args.sampler_lengths[0] = 20
    args.small_ds = True
    ds = build_fscd(split, args)

    list_dataset = []
    for vid in [59,42,10,21,39]:
        vid = vid%len(ds)
        data = ds[vid]
        list_dataset.append([f'idx{vid}//', data['imgs'], data['exemplar'][0]])
    return list_dataset


class ListImgDataset(Dataset):
    def __init__(self, base_path, img_list, exemplar_bb) -> None:
        super().__init__()
        self.base_path = base_path
        self.img_list = img_list
        self.exemplar = exemplar_bb
        self.e = None

        '''
        common settings
        '''
        self.img_height = 704   # 800
        self.img_width = 1216   # 1536
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def load_img_from_file(self, fpath_or_ndarray):
        if isinstance(fpath_or_ndarray, str):
            # bb as a box coordinates array [x1,y1,x2,y2]
            bb = self.exemplar
            ori_img = cv2.imread(os.path.join(self.base_path, fpath_or_ndarray))
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            img = F.normalize(F.to_tensor(ori_img), self.mean, self.std)
            r = min(800/img.shape[1], 800/img.shape[2])
            if self.e is None:
                self.e = img[:, bb[1]:bb[3], bb[0]:bb[2]]
                self.e = F.resize(self.e, (int(self.e.shape[1]*r), int(self.e.shape[2]*r)))
            img = F.resize(img, (int(img.shape[1]*r), int(img.shape[2]*r)))
            bb = torch.tensor([(bb[0]+bb[2])//2/ori_img.shape[1], (bb[1]+bb[3])//2/ori_img.shape[0], (bb[2]-bb[0])/ori_img.shape[1], (bb[3]-bb[1])/ori_img.shape[0]])
        else:
            # bb as a Tensor
            img = fpath_or_ndarray
            if self.e is None:
                self.e = self.exemplar
            ori_img = np.array(img.permute(1,2,0))*self.std +self.mean
            bb=None
        assert ori_img is not None
        return ori_img, img, self.e, bb

    def __len__(self):
        # return 10 # fast eval
        return len(self.img_list)

    def __getitem__(self, index):
        ori_img, img, exemplar, bb = self.load_img_from_file(self.img_list[index])
        return ori_img, img, exemplar, bb



class Detector(object):
    def __init__(self, args, model, dataset):
        self.args = args
        self.gmot = model
        self.dataset = dataset  # list of tuples: (/path/to/MOT/vidname, )

        self.predict_path = os.path.join(self.args.output_dir, 'predictions', args.resume[:-4])
        os.makedirs(self.predict_path, exist_ok=True)

    @torch.no_grad()
    def detect(self, video=0):
        v_name = self.dataset[video][0].split('/')[-3]
        if os.path.exists(str(self.predict_path)+f'/{v_name}.txt'):
            print(v_name, 'already processed')
            return
        self.gmot.track_base.clear()

        loader = DataLoader(ListImgDataset(*self.dataset[video]), 1, num_workers=2)
        lines = []
        track_instances = None
        for i, (ori_img, cur_img, exemplar, bb) in enumerate(tqdm(loader)):
            ori_img, cur_img, exemplar = ori_img[0].numpy()[:,:,::-1], cur_img[0], exemplar[0]
            # predict
            cur_img, exemplar, bb = cur_img.to(self.args.device), exemplar.to(self.args.device), bb.to(self.args.device)

            seq_h, seq_w, _ = ori_img.shape

            # predict & keep > thresh
            track_instances = self.gmot.inference_single_image(cur_img, (seq_h, seq_w), track_instances, exemplar, bb)

            # save predictions
            ori_img = (ori_img-ori_img.min())/(ori_img.max()-ori_img.min())*255
            show = (i%(len(loader)//5)==0 or i in list(range(50,60))) and self.args.debug
            lines += visualize_pred(track_instances, ori_img, self.predict_path, f'vid{v_name}_fr{i}', i, self.args.prob_detect, show)            

        with open(os.path.join(self.predict_path, f'{v_name}.txt'), 'w') as f:
            f.writelines(lines)
        print("{}: totally {} dts [{} per frame]".format(v_name, len(lines), len(lines)/len(loader)))


def load_for_eval(args):
    ARCHITECTURE = ['use_expanded_query', 'concatenate_exemplar', 'use_bmn', 'meta_arch', 'with_box_refine', 'two_stage', 'accurate_ratio', 'num_anchors', 'backbone', 'enable_fpn',  'position_embedding', 'num_feature_levels', 'enc_layers', 'dim_feedforward', 'num_queries', 'hidden_dim', 'dec_layers',  'nheads', 'enc_n_points', 'dec_n_points', 'decoder_cross_self', 'extra_track_attn', 'loss_normalizer']
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'args' in checkpoint:
        old_args = checkpoint['args']
        for k in ARCHITECTURE:
            if k not in old_args.__dict__:continue
            args.__dict__[k] = old_args.__getattribute__(k)
    
    msg = f'loading {args.resume},:     {checkpoint["args"].meta_arch} {checkpoint["args"].dec_layers}         epochs: {checkpoint["epoch"]}'
    print(msg)
    model, _, _ = build_model(args)
    model.to(args.device).eval()

    model.load_state_dict(checkpoint['model'], strict=False)

    with open(args.output_dir+'/modelstats.txt', 'a') as fout:
        fout.write(msg + '   nparams:' + str(sum(p.numel() for p in model.parameters()))+ '\n')
    return model


exemplar_n={
    'insect-1':2,
    'airplane-3':3,
    'boat-0':1, 
    'person-3':11, 
    'boat-2':5, 
    'airplane-1':0, 
    'ball-0':8, 
    'insect-3':7, 
    'boat-1':6, 
    'balloon-2':0, 
    'balloon-1':3, 
    'person-0':7, 
    'car-2':0, 
    'car-1':4, 
    'car-0':3, 
    'stock-2':6, 
    'airplane-2':0, 
    'bird-3':4, 
    'person-2':10, 
    'person-1':0, 
    'insect-0':1, 
    'stock-3':11, 
    'ball-2':8, 
    'car-3':7, 
    'boat-3':4, 
    'fish-0':3, 
    'airplane-0':10, 
    'stock-1':7, 
    'stock-0':3, 
    'bird-0':2, 
    'ball-1':0, 
    'bird-2':3, 
    'bird-1':8, 
    'fish-1':2, 
    'balloon-0':5, 
    'insect-2':10, 
    'balloon-3':8, 
    'ball-3':5, 
    'fish-3':0, 
    'fish-2':6 
}



if __name__=='__main__':
    main()