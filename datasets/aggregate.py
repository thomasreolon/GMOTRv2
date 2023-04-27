from torch.utils.data import Dataset
from numpy.random import rand
import numpy as np

from .coco import build as build_coco
from .synth import build as build_synth
from .fscd import build as build_fscd

def build(split, args):
    return AggregateDataset(args, split)

class AggregateDataset(Dataset):
    def __init__(self, args, split, phases=None, len=2000):
        self.args = args
        self.phases = args.phases if phases is None else phases
        self.datasets = []
        self._len = len

        # Add datasets
        if args.coco_path is not None:
            self.datasets.append(build_coco(split, args, full_dataset=True))
        if args.synth_path is not None:
            self.datasets.append(build_synth(split, args))
        if args.fscd_path is not None:
            self.datasets.append(build_fscd(split, args))

        # Get indices
        self.set_epoch(0)

    def set_epoch(self, epoch):
        k = max(*[x for x in self.phases.keys() if x <= epoch], -1)
        self.indices = []
        for i, dataset in enumerate(self.datasets):
            self.indices += [(i, j) for j in range(len(dataset)) if rand()<self.phases[k][i]]
        self.current_epoch = epoch
        if len(self.indices) < self._len:
            self.indices += self.indices[:self._len-len(self.indices)]
        elif len(self.indices) > self._len:
            rand_permutation = np.random.permutation(len(self.indices))[:self._len]
            self.indices = [self.indices[i] for i in rand_permutation]


    def step_epoch(self):
        self.current_epoch += 1
        self.set_epoch(self.current_epoch)

    @staticmethod
    def collate_fn(batch):
        """Forces collate a batch 1 element at a time"""
        assert len(batch) == 1, "Batch size must be 1"
        return batch[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx, fail=2):
        idx = idx % len(self) # set_epoch will change the length of the dataset
        try:
            # Get data
            dataset_idx, data_idx = self.indices[idx]
            data = self.datasets[dataset_idx][data_idx]

            # Return
            return data
        except Exception as e:
            if fail>0:
                print('err', e)
                return self.__getitem__(idx+1, fail=fail-1)
            else:
                raise e
