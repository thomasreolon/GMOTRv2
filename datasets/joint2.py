import torch
from .fscd import build as build_fscd
from .synth import SynthData
from .gmot import build as build_gmot

class JOINTFSCD(torch.utils.data.Dataset):
    def __init__(self, args, datasets) -> None:
        super().__init__()

        self.datasets = datasets
        self.args = args
        
        lens = [len(d) for d in self.datasets]
        for i in range(1,len(lens)):
            lens[i] += lens[i-1]
        self.lens = lens

    def __len__(self):
        return self.lens[-1]
    
    def __getitem__(self, idx):
        for d_num, d_items in enumerate(self.lens):
            if idx < d_items:
                return self.datasets[d_num] [d_items-idx]

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
