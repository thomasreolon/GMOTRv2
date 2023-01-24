import torch
import torch.nn.functional as F
import cv2, numpy as np

from .models import build_model

class BMNProposer(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = build_model()

    def forward(self, image, exemplar):
        if self.args.use_bmn==0: return None,None,None
        device = image.device
        # resize image
        w,h = 512, 512*image.shape[2]//image.shape[3]
        r = w / image.shape[2]
        image = F.adaptive_avg_pool2d(image, (h,w))

        we,he = int(exemplar.shape[3]*r), int(exemplar.shape[2]*r)
        exemplar =F.adaptive_avg_pool2d(exemplar, (he,we))

        # calculate scale
        scale = exemplar.shape[3] / image.shape[3] * 0.5 + exemplar.shape[2] / image.shape[3] * 0.5 
        scale = scale // (0.5 / 20)
        scale = scale if scale < 20 - 1 else 20 - 1
        patch = {'scale_embedding':torch.tensor([[scale]], device=device).int(), 'patches':exemplar[:,None]}

        # get proposals
        res = self.bmn(image, patch, True)
        interest = F.adaptive_avg_pool2d(res['density_map'], image.shape[-2:]) [0,0] #HW
        self.criterion.corr_hw = res['f_shape']

        # generate q_refs from proposals
        good_pixels1 = torch.zeros_like(interest).bool()
        c1 = interest[:-1, :] >= interest[1: , :]  # bigger than up
        c2 = interest[1: , :] >  interest[:-1, :]  # bigger than down
        c3 = interest[:, :-1] >= interest[:, 1: ]  # bigger than left
        c4 = interest[:, 1: ] >  interest[:, :-1]  # bigger than right

        good_pixels1[1:-1, 1:-1] = c1[1:, 1:-1] & c2[:-1, 1:-1] & c3[1:-1, 1:] & c4[1:-1, :-1]
        good_pixels2 = interest > interest.max()/2

        # print(interest.mean(), interest.std(), good_pixels1.sum(), good_pixels2.sum())
        if not (good_pixels1 & good_pixels2).any(): 
            return None, res['density_map'], res['corr_map']
        good_pixels = (good_pixels1 & good_pixels2).nonzero(as_tuple=True)
        xy = torch.stack((good_pixels[-1], good_pixels[-2]), dim=1).float()
        xy = xy / torch.tensor([w,h],device=device).view(1,2)

        bb = torch.tensor([[exemplar.shape[3]/image.shape[3],  exemplar.shape[2]/image.shape[2]]], device=device).expand(xy.shape[0], -1)


        if self.args.debug and torch.rand(1)>0.98:
            tmp = image.cpu()[0].permute(1,2,0).numpy()[:,:,::-1]
            tmp = (tmp-tmp.min()) / (tmp.max()-tmp.min())

            for coord in xy:
                x = int(coord[0]*tmp.shape[1]) 
                y = int(coord[1]*tmp.shape[0])
                tmp[y-1:y+2,x-2:x+2] = (.3,.3,.3)
                tmp[y,x] = (0,0,1.)
                tmp[y+1,x+1] = (0,0,1.)
                tmp[y-1,x+1] = (0,0,1.)
                tmp[y-1,x-1] = (0,0,1.)
                tmp[y+1,x-1] = (0,0,1.)
            tmp2 = interest.clone().detach().cpu()[:,:,None].expand(-1,-1,3).numpy()
            tmp2 = (tmp2-tmp2.min()) / (tmp2.max()-tmp2.min())
            tmp = np.concatenate((tmp,tmp2), axis=1)
            # cv2.imshow('ddd', tmp)

            cv2.imwrite(f'{self.args.output_dir}/debug/BMN_{len(xy)%5}.jpg', np.uint8(tmp*255))


        return torch.cat((xy,bb),dim=1), res['density_map'], res['corr_map']   # Nx4







def build(args, load_pretrained):
    model = BMNProposer(args)

    if args.bmn_pretrained and load_pretrained:
        checkpoint = torch.load(args.bmn_pretrained, map_location='cpu')
        model.model.load_state_dict(checkpoint['model'])

    return model
