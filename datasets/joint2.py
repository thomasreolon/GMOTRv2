import torch
from .fscd import build as build_fscd
from .synth import SynthData
from .gmot import build as build_gmot

class JOINTFSCD(torch.utils.data.Dataset):
    def __init__(self, args, datasets) -> None:
        super().__init__()

        self.datasets = datasets
        self.args = args
        self.lens = [len(d) for d in datasets]

    def __len__(self):
        return sum(self.lens)
    
    def __getitem__(self, idx):
        try:
            for dataset in self.datasets:
                d_len = len(dataset)
                if idx < d_len:
                    return self._check(dataset[idx])
                idx -= d_len
        except Exception as e:
            print(e, force=True)
        return dataset[idx-1] #should never happen

    def _check(self, tmp):
        if any([len(t.boxes)==0 for t in tmp['gt_instances']]):
            return self[int(torch.rand(1)*sum(self.lens))]
        return tmp


    def set_epoch(*a,**b):pass

def build(split, args):
    datasets = []
    for d_name in args.dataset_file.split('_')[1:]:
        if 'gmot' in d_name:
            datasets.append(build_gmot(split, args))
        elif 'fscd' in d_name:
            datasets.append(build_fscd(split, args))
        elif 'synth' in d_name:
            datasets.append(SynthData(args))

    return JOINTFSCD(args, datasets)
