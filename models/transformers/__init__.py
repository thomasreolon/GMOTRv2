
from .original import pos2posemb, build_deforamble_transformer as build_original


def build_deforamble_transformer(args):
    if args.meta_arch == 'deformable_detr':
        return build_original(args)
    else:
        raise NotImplementedError(args.meta_arch)
