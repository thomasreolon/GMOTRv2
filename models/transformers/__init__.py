
def build_deforamble_transformer(args):

    ## Deformable Attention
    if args.meta_arch == 'deformable_detr':
        from .original import build_deforamble_transformer
        return build_deforamble_transformer(args)

    ## different attention matrix
    if args.meta_arch == 'deformable_lowdimk':
        from .lowdimk import build_deforamble_transformer
        return build_deforamble_transformer(args)

    ## Deformable attention with custom value_projection for exemplars
    if args.meta_arch == 'deformable_detr2':
        from .almostoriginal import  build_deforamble_transformer
        return build_deforamble_transformer(args)
    else:
        raise NotImplementedError(args.meta_arch)
