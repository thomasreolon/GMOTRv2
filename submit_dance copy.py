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
from util.tool import load_model
from main import get_args_parser

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from configs.defaults import get_args_parser
from datasets.fscd import build as build_fscd

from util.plot_utils import visualize_pred, visualize_gt, train_visualize_pred

def main():
    # Info about code execution
    args = get_args_parser().parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build Model
    model = load_for_eval(args).to(args.device)
    model.track_base.score_thresh = args.prob_detect
    model.track_base.filter_score_thresh = args.prob_detect*.8
    model.track_base.miss_tolerance = 10
    model.track_embed.score_thr = args.prob_detect*.8

    # Load dataset
    print('loading dataset...')
    dataset = load_svdataset(args.dataset_file, 'train', args)

    # rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    # ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    # dataset = dataset[rank::ws]

    # Track
    det = Detector(args, model, dataset)
    print('tracking..')
    for vid in range(len(dataset)):
        det.detect(vid)

def load_svdataset(datasetname, split, args):
    assert datasetname in {'e2e_gmot', 'e2e_fscd'}, f'invalid dataset "{datasetname}"'
    assert split in {'train', 'val', 'test'}, f'invalid dataset "{split}"'
    
    if datasetname=='e2e_gmot':
        return load_gmot(split, args)
    elif datasetname=='e2e_fscd':
        return load_fscd(split, args)

def load_gmot(split, args):
    base_path = args.gmot_dir+'/GenericMOT_JPEG_Sequence/'

    list_dataset = []
    videos = os.listdir(base_path)

    for video in videos:
        # get 1st BB
        gt = args.gmot_dir+f'/track_label/{video}.txt'
        with open(gt, 'r') as fin:
            for _ in range(300):
                line = fin.readline()
                if line[0] == '0': break
        line = [int(l) for l in line.split(',')]
        bb = line[2], line[3], line[2]+line[4], line[3]+line[5], 

        # get images
        imgs = sorted(os.listdir(f'{base_path}{video}/img1'))

        list_dataset.append((f'{base_path}{video}/img1/', imgs, bb))  # none should be the exemplar_bb xyxy

    return list_dataset


def load_fscd(split, args):
    args.sampler_lengths[0] = 20
    args.small_ds = True
    ds = build_fscd(split, args)

    list_dataset = []
    for vid in [59,42,10,21,39]:
        data = ds[vid]
        list_dataset.append([None, data['imgs'], data['exemplar'][0]])
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
            frame = F.normalize(F.to_tensor(ori_img).permute(2,0,1), self.mean, self.std)
            if self.e is None:
                self.e = frame[:, bb[1]:bb[3], bb[0]:bb[2]]
        else:
            # bb as a Tensor
            img = fpath_or_ndarray
            if self.e is None:
                self.e = self.exemplar
            ori_img = np.array(img.permute(1,2,0))*self.std +self.mean
        assert ori_img is not None
        return ori_img, img, self.e

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        ori_img, img, exemplar = self.load_img_from_file(self.img_list[index])
        return ori_img, img, exemplar



class Detector(object):
    def __init__(self, args, model, dataset):
        self.args = args
        self.gmot = model
        self.dataset = dataset  # list of tuples: (/path/to/MOT/vidname, )

        self.predict_path = os.path.join(self.args.output_dir, 'predictions')
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]
    
    @staticmethod
    def fix(xyxy, shape):
        H,W,_ = shape
        clean = lambda x,X: max(0,min(x,X))
        x1,y1,x2,y2 = xyxy
        return (clean(x1,W), clean(y1,H), clean(x2,W), clean(y2,H))

    @torch.no_grad()
    def detect(self, video=0):
        self.gmot.track_base.clear()
        self.gmot.eval()

        loader = DataLoader(ListImgDataset(*self.dataset[video]), 1, num_workers=2)
        lines = []
        track_instances = None
        for i, (ori_img, cur_img, exemplar) in enumerate(tqdm(loader)):
            ori_img, cur_img, exemplar = ori_img[0].numpy(), cur_img[0].to(self.args.device), exemplar[0].to(self.args.device)
            data_dict = {'imgs':[cur_img], 'exemplar':[exemplar],'gt_instances':[None]}
            if i==0: visualize_gt(data_dict, self.predict_path, i=video)


            # predict & keep > thresh
            outputs = self.gmot(data_dict, track_instances)
            track_instances = outputs['dt_instances'] = outputs['track_instances']
             

            lines += train_visualize_pred(data_dict, outputs, self.predict_path, self.args.prob_detect, video)

            # # save predictions
            # seq_h, seq_w, _ = ori_img.shape
            # ori_img = (ori_img-ori_img.min())/(ori_img.max()-ori_img.min())*255
            # show = i%(len(loader)//5)==0 and self.args.debug
            # lines += visualize_pred(track_instances, ori_img, self.predict_path, f'vid{video}_fr{i}', i, self.args.prob_detect, show)            

        with open(os.path.join(self.predict_path, f'{video}.txt'), 'w') as f:
            f.writelines(lines)
        print("totally {} dts".format(len(lines)))

    # def detect(self, video=0):
    #     self.gmot.track_base.clear()

    #     loader = DataLoader(ListImgDataset(*self.dataset[video]), 1, num_workers=2)
    #     lines = []
        # track_instances = None
        # for i, (ori_img, cur_img, exemplar) in enumerate(tqdm(loader)):
        #     ori_img, cur_img, exemplar = ori_img[0].numpy(), cur_img[0], exemplar[0]
        #     if i==0: visualize_gt({'imgs':[cur_img], 'exemplar':[exemplar],'gt_instances':[None]}, self.predict_path, i=video)
        #     # predict
        #     cur_img, exemplar = cur_img.to(self.args.device), exemplar.to(self.args.device)

        #     seq_h, seq_w, _ = ori_img.shape

        #     # predict & keep > thresh
        #     track_instances = self.gmot.inference_single_image([cur_img], (seq_h, seq_w), track_instances, [exemplar])

        #     # save predictions
        #     ori_img = (ori_img-ori_img.min())/(ori_img.max()-ori_img.min())*255
        #     show = i%(len(loader)//5)==0 and self.args.debug
        #     lines += visualize_pred(track_instances, ori_img, self.predict_path, f'vid{video}_fr{i}', i, self.args.prob_detect, show)            

    #     with open(os.path.join(self.predict_path, f'{video}.txt'), 'w') as f:
    #         f.writelines(lines)
    #     print("totally {} dts".format(len(lines)))


def load_for_eval(args):
    ARCHITECTURE = ['meta_arch', 'with_box_refine', 'two_stage', 'accurate_ratio', 'num_anchors', 'backbone', 'enable_fpn',  'position_embedding', 'num_feature_levels', 'enc_layers', 'dim_feedforward', 'num_queries', 'hidden_dim', 'dec_layers',  'nheads', 'enc_n_points', 'dec_n_points', 'decoder_cross_self', 'extra_track_attn', 'loss_normalizer']
    model_path = (os.path.exists(args.resume) and args.resume) or args.output_dir + '/checkpoint.pth'

    print("loading... ", model_path)
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'args' in checkpoint:
        old_args = checkpoint['args']
        for k in ARCHITECTURE:
            args.__dict__[k] = old_args.__getattribute__(k)

    from models import build_model
    model,_,_ = build_model(args)
    load_model(model, model_path)
    return model.eval()


if __name__=='__main__':
    main()