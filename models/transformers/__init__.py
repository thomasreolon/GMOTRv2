
def build_deforamble_transformer(args):
    if args.meta_arch == 'deformable_detr':
        from .original import build_deforamble_transformer as build_original
        return build_original(args)
    if args.meta_arch == 'deformable_lowdimk':
        from .lowdimk import build_deforamble_transformer as build_lowdimk
        return build_lowdimk(args)
    if args.meta_arch == 'deformable_attn':
        from .attn import  build_deforamble_transformer as build_attn
        return build_attn(args)
    if args.meta_arch == 'deformable_attnkernel':
        from .attn_kernel import  build_deforamble_transformer as build_attn2
        return build_attn2(args)
    if args.meta_arch == 'deformable_mixed':
        from .mixed import  build_deforamble_transformer as build_mixed
        return build_mixed(args)
    if args.meta_arch == 'deformable_anchor':
        from .anchor import  build_deforamble_transformer as build_anchor
        return build_anchor(args)
    if args.meta_arch == 'deformable_detr_full':
        from .original_full import  build_deforamble_transformer as build_full
        return build_full(args)
    else:
        raise NotImplementedError(args.meta_arch)
