import motmetrics as mm
import pandas as pd
import os
import numpy as np

def compute_mota(output_path, fname, gt_folder, preds_folder):
    gt_files = os.listdir(gt_folder)
    summaries, names = [], []
    for file in os.listdir(preds_folder):
        if file not in gt_files: continue
        print('evaluating...', file)

        summary = motMetricsEnhancedCalculator(gt_folder+file, preds_folder+file)        
        summaries.append(summary)
        names.append(file.split('.')[0])

    stats = pd.concat(summaries)
    stats['video'] = names
    avg = stats.mean(axis=0)

    os.makedirs(f'{output_path}/results', exist_ok=True)
    avg[['mota', 'idf1', 'motp', 'recall', 'precision', 'num_misses']].to_csv(f'{output_path}/results/avg{fname}.csv')
    stats[['video', 'mota', 'idf1', 'motp', 'recall', 'precision', 'num_misses']].to_csv(f'{output_path}/results/{fname}.csv')



def read_mot_results(filename):
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')

                fid = int(linelist[0])
                results_dict.setdefault(fid, list())

                score = 1
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict

def motMetricsEnhancedCalculator(gtSource, tSource):  
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=',')

    # load tracking output
    t = np.loadtxt(tSource, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:,0].max())):
        frame += 1 # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
        t_dets = t[t[:,0]==frame,1:6] # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                    max_iou=0.5) # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:,0].astype('int').tolist(), \
                t_dets[:,0].astype('int').tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                        'recall', 'precision', 'num_objects', \
                                        'mostly_tracked', 'partially_tracked', \
                                        'mostly_lost', 'num_false_positives', \
                                        'num_misses', 'num_switches', \
                                        'num_fragmentations', 'mota', 'motp' \
                                        ], \
                        name='acc')

    return summary


if __name__=='__main__':
    main()