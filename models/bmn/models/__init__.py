import torch
import pathlib


from .backbone import build_backbone
from .counter import get_counter
from .epf_extractor import build_epf_extractor
from .refiner import build_refiner
from .matcher import build_matcher
from .class_agnostic_counting_model import CACModel
from ..config import cfg

def build_model():
    cfg_path = pathlib.Path(__file__).parent.resolve() / '../config/test_bmnet+.yaml'
    cfg.merge_from_file(cfg_path)

    backbone = build_backbone(cfg)
    epf_extractor = build_epf_extractor(cfg)
    refiner = build_refiner(cfg)
    matcher = build_matcher(cfg)
    counter = get_counter(cfg)
    model = CACModel(backbone, epf_extractor, refiner, matcher, counter, cfg.MODEL.hidden_dim)

    return model
