import torch
from .fscd import build as build_fscd
from .gmot import build as build_gmot
from .synth import SynthData

class JOINTFSCD(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.fscd = build_fscd('train', args)
        self.synth = SynthData(args)
        self.gmot = build_gmot('all', args)
        self.lens = (len(self.fscd), len(self.fscd)+len(self.synth), len(self.fscd)+len(self.synth)+len(self.gmot))

    def __len__(self):
        return self.lens[-1]
    
    def __getitem__(self, idx):
        if idx < self.lens[0]:
            return self.fscd[idx]
        elif idx < self.lens[1]:
            return self.synth[idx-self.lens[0]]
        else:
            return self.gmot[idx-self.lens[1]]

    def set_epoch(*a,**b):pass

def build(split, args):
    return JOINTFSCD(args)
