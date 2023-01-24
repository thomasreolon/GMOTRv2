import torch
import torch.nn.functional as F
import cv2, numpy as np
import math

from .models import build_model
from util.misc import is_main_process

class BMNProposer(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = build_model()
        self.g_blur = None
        self.g_blur2 = None

    def forward(self, image, exemplar, gt_boxes=None): # TODO: benchmark time exe
        device = image.device
        # resize image
        w,h = 512, 512*image.shape[2]//image.shape[3]
        r = w / image.shape[2]
        image = F.adaptive_avg_pool2d(image, (h,w))

        w,h = int(exemplar.shape[3]*r), int(exemplar.shape[2]*r)
        exemplar =F.adaptive_avg_pool2d(exemplar, (h,w))

        # calculate scale
        scale = exemplar.shape[3] / image.shape[3] * 0.5 + exemplar.shape[2] / image.shape[3] * 0.5 
        scale = scale // (0.5 / 20)
        scale = scale if scale < 20 - 1 else 20 - 1
        patch = {'scale_embedding':torch.tensor([[scale]], device=device).int(), 'patches':exemplar[:,None]}

        # get proposals
        res = self.model(image, patch, True)
        interest = res['density_map'][0,0] #HW
        loss, count_map = self._loss(interest, res['corr_map'], gt_boxes)

        # generate q_refs from proposals
        good_pixels1 = torch.zeros_like(interest).bool()
        c1 = interest[:-1, :] >= interest[1: , :]  # bigger than up
        c2 = interest[1: , :] >  interest[:-1, :]  # bigger than down
        c3 = interest[:, :-1] >= interest[:, 1: ]  # bigger than left
        c4 = interest[:, 1: ] >  interest[:, :-1]  # bigger than right

        good_pixels1[1:-1, 1:-1] = c1[1:, 1:-1] & c2[:-1, 1:-1] & c3[1:-1, 1:] & c4[1:-1, :-1]
        good_pixels2 = interest > interest.max()/3

        ref_pts = None
        if (good_pixels1 & good_pixels2).any(): 
            good_pixels = (good_pixels1 & good_pixels2).nonzero(as_tuple=True)
            xy = torch.stack((good_pixels[-1], good_pixels[-2]), dim=1).float()
            h,w = interest.shape
            xy = xy / torch.tensor([w,h],device=device).view(1,2)

            bb = torch.tensor([[exemplar.shape[3]/image.shape[3],  exemplar.shape[2]/image.shape[2]]], device=device).expand(xy.shape[0], -1)
            ref_pts = torch.cat((xy,bb),dim=1)
        
            if self.args.debug and torch.rand(1)>0.98:
                self._debug_visualization(image, xy, interest, res['corr_map'], count_map)

        return ref_pts, loss   # Nx4

    def _loss(self, density_map, corr_map, boxes):
        # GT heatmap
        with torch.no_grad():
            count_map = torch.zeros_like(density_map)
            h,w = density_map.shape
            boxes = (boxes * torch.tensor([w,h,w,h], device=boxes.device)[None]).int().cpu().tolist()
            for (x,y,w,h) in boxes:
                self.apply_blur(count_map, x,y,w,h)
        
        # density loss
        density_loss = F.mse_loss(density_map, count_map)
        
        # correlation loss
        corr_hw = density_map.shape[0]//16, density_map.shape[1]//16
        corr_map = corr_map.view(corr_hw)
        sm_count_map = F.adaptive_avg_pool2d(count_map[None], corr_hw).view(corr_hw)
        positive = sm_count_map > sm_count_map.max()*.8
        corr_map = (corr_map-corr_map.mean()) / (corr_map.std()+1e-8)
        corr_map = corr_map.exp()
        corr_loss = - ( corr_map[positive].sum() / (corr_map.sum()+1e-10) ).log()
        # corr_loss = 0

        return corr_loss*1e-5 + density_loss, count_map

    def apply_blur(self, count_map, x,y,w,h):
        if self.g_blur is None:
            x_coord = torch.arange(64)
            x_grid = x_coord.repeat(64).view(64, 64)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
            mean = (64 - 1)/2.
            gaussian_kernel = torch.exp(
                                -torch.sum((xy_grid - mean)**2., dim=-1) / (2*64)
                            )
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
            self.g_blur = gaussian_kernel.to(count_map.device)

        kernel = self.g_blur.to(count_map.device)
        kernel = F.adaptive_avg_pool2d(kernel[None], (h,w))[0]
        x1 = x-kernel.shape[1]//2
        y1 = y-kernel.shape[0]//2
        x2 = x+kernel.shape[1] -(kernel.shape[1]//2)
        y2 = y+kernel.shape[0] -(kernel.shape[0]//2)

        r_x1 = max(0,x1)
        r_y1 = max(0,y1)
        r_x2 = min(count_map.shape[1]-1,x2)
        r_y2 = min(count_map.shape[0]-1,y2)

        count_map[r_y1:r_y2, r_x1:r_x2] += kernel[r_y1-y1:r_y2-y2+kernel.shape[0], x1-r_x1:r_x2-x2+kernel.shape[1]]

    # def _loss_fast(self, density_map, corr_map, boxes):
    #     # density loss
    #     count_map = torch.zeros_like(density_map)
    #     std = 20 # boxes[:,2:].mean().item() * min(count_map.shape)
    #     blurrer = self.get_gauss_blur(std).to(count_map.device)
    #     for bb in boxes:
    #         x, y = int(bb[0]*density_map.shape[1]), int(bb[1]*density_map.shape[0])
    #         count_map[y,x] = 1.
    #     with torch.no_grad():
    #         count_map = blurrer(count_map[None,None])[0,0]
    #         count_map = count_map - count_map.min()
    #         count_map = count_map / count_map.sum() * len(boxes)
    #     density_loss = F.mse_loss(density_map, count_map)
        
    #     # correlation loss
    #     corr_hw = density_map.shape[0]//16, density_map.shape[1]//16
    #     corr_map = corr_map.view(corr_hw)
    #     sm_count_map = F.adaptive_avg_pool2d(count_map[None], corr_hw).view(corr_hw)
    #     positive = sm_count_map > sm_count_map.max()*.8
    #     corr_map = (corr_map-corr_map.mean()) / (corr_map.std()+1e-8)
    #     corr_map = corr_map.exp()
    #     corr_loss = - ( corr_map[positive].sum() / (corr_map.sum()+1e-10) ).log()
    #     # corr_loss = 0

    #     return corr_loss*1e-5 + density_loss, count_map

    # def get_gauss_blur(self, std=14, kernel_size=32):
    #     x_coord = torch.arange(kernel_size)
    #     x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    #     y_grid = x_grid.t()
    #     xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    #     if self.g_blur2 is None:
    #         mean = (kernel_size - 1)/2.

    #         # Calculate the 2-dimensional gaussian kernel which is
    #         gaussian_kernel = (1./(2.*math.pi*std)) *\
    #                         torch.exp(
    #                             -torch.sum((xy_grid - mean)**2., dim=-1) /\
    #                             (2*std)
    #                         )
    #         # Make sure sum of values in gaussian kernel equals 1.
    #         gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    #         # Reshape to 2d depthwise convolutional weight
    #         gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    #         gaussian_filter = torch.nn.Conv2d(in_channels=1, out_channels=1,
    #                                     kernel_size=kernel_size, padding='same', bias=False)

    #         gaussian_filter.weight.data = gaussian_kernel
    #         gaussian_filter.weight.requires_grad = False
    #         self.g_blur2 = gaussian_filter.float()
    #     return self.g_blur2

    def _debug_visualization(self, image, xy, interest, corr_map, count_map):
        
        #-------original image W proposals
        tmp = F.adaptive_avg_pool2d(image, interest.shape)
        tmp = tmp.cpu()[0].permute(1,2,0).numpy()[:,:,::-1]
        tmp = tmp / 4 + .5  # special normalization
        for coord in xy:
            x = int(coord[0]*tmp.shape[1]) 
            y = int(coord[1]*tmp.shape[0])
            tmp[y-1:y+2,x-2:x+2] = (.9,.9,.9)
            tmp[y,x] = (0,0,1.)
            tmp[y+1,x+1] = (0,0,1.)
            tmp[y-1,x+1] = (0,0,1.)
            tmp[y-1,x-1] = (0,0,1.)
            tmp[y+1,x-1] = (0,0,1.)


        #-------density map
        tmp2 = interest.clone().detach().cpu()[:,:,None].expand(-1,-1,3).numpy()
        tmp2 = (tmp2-tmp2.min()) / (tmp2.max()-tmp2.min())

        #-------count map
        tmp3 = count_map.cpu()[:,:,None].expand(-1,-1,3).numpy()
        tmp3 = (tmp3-tmp3.min()) / (tmp3.max()-tmp3.min())

        #-------correlation
        tmp4 = corr_map.view(1,1,interest.shape[0]//16, interest.shape[1]//16)
        tmp4 = tmp4.clone().detach().cpu()
        tmp4 = F.adaptive_avg_pool2d(tmp4, interest.shape)
        tmp4 = tmp4[0,0,:,:,None].expand(-1,-1,3).numpy()
        tmp4 = (tmp4-tmp4.min()) / (tmp4.max()-tmp4.min())

        row1 = np.concatenate((tmp,tmp4), axis=1)
        row2 = np.concatenate((tmp2,tmp3), axis=1)
        tmp = np.concatenate((row1, row2), axis=0)
        cv2.imwrite(f'{self.args.output_dir}/debug/BMN_{len(xy)%5}.jpg', np.uint8(tmp*255))


def build(args):
    model = BMNProposer(args)

    if args.bmn_pretrained and is_main_process():
        checkpoint = torch.load(args.bmn_pretrained, map_location='cpu')
        model.model.load_state_dict(checkpoint['model'])

    return model
