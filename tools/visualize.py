# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


from collections import defaultdict
from glob import glob
import json
import os
import cv2
import subprocess
from tqdm import tqdm
import numpy as np

def get_color(i):
    return [(i * 23 * j + 43) % 255 for j in range(3)]


def process(trk_path, img_list, output="output.mp4"):
    h, w, _ = cv2.imread(img_list[0]).shape
    command = [
        "/usr/bin/ffmpeg",
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{w}x{h}',  # size of one frame
        '-pix_fmt', 'bgr24',
        '-r', '20',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-s', f'{w//2*2}x{h//2*2}',
        '-an',  # Tells FFMPEG not to expect any audio
        '-loglevel', 'error',
        '-crf', '26',
        '-pix_fmt', 'yuv420p'
    ]
    writing_process = subprocess.Popen(command + [output], stdin=subprocess.PIPE)

    tracklets = defaultdict(list)
    for line in open(trk_path):
        t, id, *xywhs = line.split(',')[:7]
        t, id = map(int, (t, id))
        x, y, w, h, s = map(float, xywhs)
        tracklets[t].append((id, *map(int, (x, y, x+w, y+h))))

    for i, path in enumerate(tqdm(sorted(img_list))):
        im = cv2.imread(path)
        for det in det_db[path.replace('.jpg', '.txt')]:
            x1, y1, w, h, _ = map(int, map(float, det.strip().split(',')))
            im = cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 255, 255), 6)
        for j, x1, y1, x2, y2 in tracklets[i + 1]:
            im = cv2.rectangle(im, (x1, y1), (x2, y2), get_color(j), 4)
            im = cv2.putText(im, f"{j}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, get_color(j), 2)
        writing_process.stdin.write(im.tobytes())

def process2(trk_path, img_path, output="output.mp4"):
    H,W = 720,1080
    command = [
        "/usr/bin/ffmpeg",
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{W}x{H}',  # size of one frame
        '-pix_fmt', 'bgr24',
        '-r', '20',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-an',  # Tells FFMPEG not to expect any audio
        '-loglevel', 'error',
        '-crf', '26',
        '-pix_fmt', 'yuv420p'
    ]
    writing_process = subprocess.Popen(command + [output], stdin=subprocess.PIPE)
    for file in [x for x in os.listdir(trk_path) if '.txt' in x]:
        annot_path = f'{trk_path}/{file}'
        img_list = os.listdir(f'{img_path}/{file[:-4]}/img1/')

        tracklets = defaultdict(list)
        for line in open(annot_path):
            t, id, *xywhs = line.split(',')[:7]
            t, id = map(int, (t, id))
            x, y, w, h, s = map(float, xywhs)
            tracklets[t].append((id, *map(int, (x, y, x+w, y+h))))

        for i, path in enumerate(tqdm(sorted(img_list))):
            im = cv2.imread(f'{img_path}/{file[:-4]}/img1/{path}')
            for j, x1, y1, x2, y2 in tracklets[i]:
                im = cv2.rectangle(im, (x1, y1), (x2, y2), get_color(j), 4)
                im = cv2.putText(im, f"{j}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, get_color(j), 2)
            

            r_diff = W/H - im.shape[1]/im.shape[0]
            if r_diff < 0:
                # add pad top
                pad = int(im.shape[1] * -r_diff)
                pad = 100*np.ones_like(im)[:pad]
                im = np.concatenate((im, pad), axis=0)
            elif r_diff > 0:
                # add pad top
                pad = int(im.shape[1] * r_diff)
                pad = 100*np.ones_like(im)[:, :pad]
                im = np.concatenate((im, pad), axis=1)

            im = cv2.resize(im, (W,H))
            writing_process.stdin.write(im.tobytes())

