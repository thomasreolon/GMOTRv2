import os, cv2, numpy as np
import torch
from models.structures import Instances
import torchvision.transforms.functional as F

from scipy.ndimage import rotate
from.transforms import MotRandomResize

def add_patch(img, patch, coord):
    h,w,_ = patch.shape
    x = max(0,min(coord[0], img.shape[1]-w-1))
    y = max(0,min(coord[1], img.shape[0]-h-1))
    img[y:y+h, x:x+w] = patch
    coord[0]=x
    coord[1]=y

def get_movement(img, strength):
    dw = int((torch.rand(1)-.5)*img.shape[1]*strength)
    dh = int((torch.rand(1)-.5)*img.shape[0]*strength)
    return dw, dh

def augment(patch, img_h=999, img_w=999):
    r = torch.rand(4)
    if r[0]>.0: # rotate
        patch = rotate_img(patch, float(r[1]*4-2))
    if r[2]>.0: #resize
        w,h = int(patch.shape[1]*(.86+.28*r[3])), int(patch.shape[0]*(.9+.2*r[3]))
        w,h = max(w,16), max(h,16)
        w,h = min(img_w//4,w), min(img_h//4,h)
        patch = cv2.resize(patch, (w,h))
    return patch

def simulate(img, data):
    gt_bb = []
    gt_idx = []
    for i in range(len(data)):
        patch, coord, velocity, idx = data[i]
        # draw box
        if coord is not None and idx>=0:
            add_patch(img, patch, coord)
            bb = [coord[0]+patch.shape[1]/2, coord[1]+patch.shape[0]/2, patch.shape[1], patch.shape[0]]
            gt_bb.append([int(b) for b in bb])
            gt_idx.append(idx)
        
        # update position
        if coord is None:
            coord = [int(torch.rand(1)*img.shape[1]), int(torch.rand(1)*img.shape[0])]
            data[i][1] = coord
        coord[0] += int(velocity[0])
        coord[1] += int(velocity[1])
        velocity[0] *= .8 + torch.rand(1)*.4
        velocity[1] *= .8 + torch.rand(1)*.4
        patch = augment(patch, *img.shape[:2])
        data[i][0] = patch

        if torch.rand(1)>.5 and (min(coord)==0 or coord[0]>img.shape[1]-20 or coord[1]>img.shape[0]-20):
            data[i][3] = -1
    
    return img, gt_bb, gt_idx



class SynthData(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        dir_path = args.synth_path
        self.transf = MotRandomResize([608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992], max_size=1536)

        samples = []
        images  = [x for x in os.listdir(dir_path) if x != 'bg']
        for folder in images:
            crops = []
            bg = cv2.imread(dir_path+'/bg/'+folder)
            for img in os.listdir(dir_path+'/'+folder):
                img = cv2.imread(dir_path+'/'+folder+'/'+img)
                crops.append(img)
            samples.append((bg, crops))
        self.samples = samples

        bgs = []
        images = [x for x in os.listdir(dir_path+'/bg/') if 'a' in x]
        for bg in images:
            img = cv2.imread(dir_path+'/bg/'+bg)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = 0.9 + 0.1*img / 254.9
            w,h = 700, int(700*img.shape[0]/img.shape[1])
            img = cv2.resize(img, (w,h))
            bgs.append(img[:,:,None])
        self.bgs = bgs

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        bg_idx = (idx*2999) % len(self.bgs)
        idx = idx % len(self.samples)
        d_idx = idx-1  - (idx*421) % (len(self.samples)-1)
        
        bg = self.bgs[bg_idx]
        base_bg, crops = self.samples[idx]
        # get a random bg with similar pattern colors to original
        a,b,c,d = (1+torch.rand(4)*min(bg.shape[:2])/5).int()
        bg = bg[a:-b, c:-d]
        base_bg = cv2.resize(base_bg, (bg.shape[1], bg.shape[0])) * bg
        _, distractions = self.samples[d_idx]
        for patch in distractions:
            r = 0.5 + 0.5*torch.rand(1)
            patch = cv2.resize(patch, (int(patch.shape[1]*r), int(patch.shape[0]*r)))
            coord = [int(torch.rand(1)*base_bg.shape[1]), int(torch.rand(1)*base_bg.shape[0])]
            add_patch(base_bg, patch, coord)
        scale = [608, 640, 672, 704, 736, 768, 800,][int(torch.rand(1)*7)]
        base_bg = cv2.resize(base_bg, (scale, int(scale*base_bg.shape[0]/base_bg.shape[1])))

        r = max((60-(len(crops)*(.6+.4*torch.rand(1)))+torch.rand(1)*40 )/2, 16)/ min(crops[0].shape[:2])
        for i in range(len(crops)):
            crops[i] = cv2.resize(crops[i], (int(crops[i].shape[1]*r), int(crops[i].shape[0]*r)))

        data = []
        base_v = get_movement(base_bg, 0.02)
        for i, patch in enumerate(crops):
            coord = [int(torch.rand(1)*base_bg.shape[1]), int(torch.rand(1)*base_bg.shape[0])]
            velocity = get_movement(base_bg, 0.1)
            if i<2: coord=None
            data.append([patch, coord, [base_v[0]+velocity[0], base_v[1]+velocity[1]], i])

        # exemplar
        exemplar = None
        # [F.normalize(F.to_tensor(patch[:,:,::-1]/255), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).float(), torch.tensor([coord[0]+patch.shape[1]/2, coord[1]+patch.shape[0]/2, patch.shape[1], patch.shape[0]])]
        # exemplar[1] = exemplar[1] / torch.tensor([base_bg.shape[1], base_bg.shape[0], base_bg.shape[1], base_bg.shape[0]])

        # simulate movement
        images = []
        gt_instances = []
        for i in range(self.args.sampler_lengths[0]):
            img, gt_bb, gt_idx = simulate(base_bg.copy(), data)
            images.append(F.normalize(F.to_tensor(img[:,:,::-1]/255), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).float())
            inst = Instances((1,1))
            inst.boxes = torch.tensor(gt_bb) / torch.tensor([[base_bg.shape[1], base_bg.shape[0], base_bg.shape[1], base_bg.shape[0]]])
            inst.obj_ids = torch.tensor(gt_idx).long()
            inst.labels = torch.zeros_like(inst.obj_ids)
            gt_instances.append(inst)

            if exemplar is None:
                exemplar = [0,0]
                x,y,w,h = gt_bb[-1]
                exemplar[0] = images[-1][:, y-h//2:y+h//2, x-w//2:x+w//2]
                exemplar[1] = inst.boxes[-1]

        return {
        'imgs': images,
        'gt_instances': gt_instances,
        'exemplar': exemplar,
    }


def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return np.uint8(img)
