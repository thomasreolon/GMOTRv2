from importlib import import_module

__all__ = [
    'coco',
    'gmot',
    'synth',
    'fscd',
    'aggregate',
]


def get_dataset(name, split, args):
    """Get a dataset by name.

    Args:
        name : str                  = Dataset name
        args : argparse.Namespace   = configs specified in configs/default.py
    """
    if name in __all__:
        return import_module('.' + name, package='datasets').build(split, args)
    else:
        raise ValueError('Cannot find dataset: {}'.format(name))



