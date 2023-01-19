# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Plotting utilities to visualize training logs.
"""
import cv2, os
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch import Tensor

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if dir.exists():
            continue
        raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


def draw_boxes(image: Tensor, boxes: Tensor, color=(0, 255, 0), texts=None) -> np.ndarray:
    if isinstance(image, Tensor):
        cv_image = image.detach().cpu().numpy()
    else:
        cv_image = image
    if isinstance(boxes, Tensor):
        cv_boxes = boxes.detach().cpu().numpy()
    else:
        cv_boxes = boxes

    tl = round(0.002 * max(image.shape[0:2])) + 1  # line thickness
    tf = max(tl - 1, 1)
    for i in range(len(boxes)):
        box = cv_boxes[i]
        x1, y1 = box[0:2]
        x2, y2 = box[2:4]
        cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color=color)
        if texts is not None:
            cv2.putText(cv_image, texts[i], (int(x1), int(y1+10)), 0, tl/3, [225, 255, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA)
    return cv_image


def draw_ref_pts(image: Tensor, ref_pts: Tensor) -> np.ndarray:
    if isinstance(image, Tensor):
        cv_image = image.detach().cpu().numpy()
    else:
        cv_image = image
    if isinstance(ref_pts, Tensor):
        cv_pts = ref_pts.detach().cpu().numpy()
    else:
        cv_pts = ref_pts
    for i in range(len(cv_pts)):
        x, y, is_pos = cv_pts[i]
        color = (0, 1, 0) if is_pos else (1, 1, 1)
        cv2.circle(cv_image, (int(x), int(y)), 2, color)
    return cv_image


def image_hwc2chw(image: np.ndarray):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return image




def _debug_frame(frame, out_w=400):
    """util to make frame to writable"""
    if len(frame.shape) == 4: frame = frame[0]
    frame = np.ascontiguousarray(frame.clone().cpu().permute(1,2,0).numpy() [:,:,::-1]) /4+0.4 # frame in BGR
    frame = np.uint8(255*(frame-frame.min())/(frame.max()-frame.min()))
    h,w,_ = frame.shape
    return cv2.resize(frame, (out_w,int(out_w*h/w)))

def visualize_gt(data_dict, output_dir, i=0):
    os.makedirs(output_dir, exist_ok=True)   
    # image shape
    num_imgs = len(data_dict['imgs']) + 1
    num_rows = int(np.sqrt(num_imgs))
    whites_to_add = 1 + num_imgs - num_rows*(num_rows+1)

    # write bboxes on images
    imgs = []
    for img, gt in zip(data_dict['imgs'], data_dict['gt_instances']):
        img = _debug_frame(img, 600)
        H,W,_ = img.shape
        def clean(x,X): return int(max(0,min(x, X-1)))
        if gt is not None:
            for box in gt.boxes:
                box = (box.view(2,2) * torch.tensor([W, H], device=box.device).view(1,2)).int()
                x1,x2 = box[0,0] - box[1,0].div(2,rounding_mode='trunc'), box[0,0] + box[1,0].div(2,rounding_mode='trunc')
                y1,y2 = box[0,1] - box[1,1].div(2,rounding_mode='trunc'), box[0,1] + box[1,1].div(2,rounding_mode='trunc')
                x1,x2,y1,y2 = clean(x1,W),clean(x2,W),clean(y1,H),clean(y2,H)
                tmp = img[y1:y2, x1:x2].copy()
                img[y1-2:y2+2, x1-2:x2+2] = (255,0,255)
                img[y1:y2, x1:x2] = tmp
        imgs.append(img)
    imgs += [200*np.ones_like(img) for _ in range(whites_to_add)]

    # add exemplar
    exe_scale = int(data_dict['exemplar'][0].shape[2] * 600 / data_dict['imgs'][0].shape[2])
    exemplar = _debug_frame(data_dict['exemplar'][0], exe_scale)
    h1,h2 = H//3, H//3 +exemplar.shape[0]
    w1,w2 = W//3, W//3 +exemplar.shape[1]
    imgs[-1][h1:h2, w1:w2] = exemplar

    # unique image
    imgs = np.stack(imgs).reshape(num_rows, num_rows+1, H,W,3)
    imgs = np.concatenate([i for i in imgs], axis=1)
    imgs = np.concatenate([i for i in imgs], axis=1)

    cv2.imwrite(f'{output_dir}/im{i}_gt.jpg', imgs)

def visualize_pred(dt_instances, img, output_dir='outputs/', f_name=None, f_num=0, prob_tresh=0.6, save=False):
    if f_name is None: f_name=f'fr{f_num}_'
    os.makedirs(output_dir, exist_ok=True)
    # filter by score
    keep = dt_instances.scores > 0.1
    dt_instances = dt_instances[keep]

    # filter by area
    wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
    keep =(wh[:, 0] * wh[:, 1]) > 30
    dt_instances = dt_instances[keep]

    lines = []
    save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
    if len(dt_instances)>0:
        bbox_xyxy = dt_instances.boxes.tolist()
        identities = dt_instances.obj_idxes.tolist()

        for xyxy, track_id in zip(bbox_xyxy, identities):
            if track_id < 0 or track_id is None:
                continue
            x1, y1, x2, y2 = [max(0, int(a)) for a in xyxy]
            w, h = x2 - x1, y2 - y1
            lines.append(save_format.format(frame=f_num, id=track_id, x1=x1, y1=y1, w=w, h=h))
            color = tuple([((5+track_id*3)*4909 % p)%256 for p in (3001, 1109, 2027)])

            tmp = img[ y1:y2, x1:x2].copy()
            img[y1-3:y2+3, x1-3:x2+3] = color
            img[y1:y2, x1:x2] = tmp
    if save:
        cv2.imwrite(f'{output_dir}/{f_name}_pred.jpg', np.uint8(img))
    return lines


def train_visualize_pred(data_dict, outputs, output_dir, prob, batch_n):
    lines = []
    track_inst = outputs['post_proc'] if 'post_proc' in outputs else outputs['post_proc']
    for i, (img, dt_inst) in enumerate(zip(data_dict['imgs'], track_inst)):
        img = _debug_frame(img, img.shape[2])
        lines += visualize_pred(dt_inst, img, output_dir, f'im{batch_n}_fr{i}', i, prob, True)
    return lines
