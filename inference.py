import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import torch
from detectron2.engine import default_setup
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize

def get_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.backbone))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.backbone)
    
    default_setup(cfg, args)
    return cfg

def get_pedestrian_bboxes(img, predictor):
    predictions = predictor(img.cpu().detach().numpy())['instances'].to('cpu')
    bboxes = predictions.pred_boxes if predictions.has('pred_boxes') else None
    scores = predictions.scores if predictions.has('scores') else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has('pred_classes') else None)
    return bboxes, scores, classes

def pre_process(
    clip,
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes)

    # Normalize images by mean and std.
    clip = normalize(clip, np.array(data_mean, dtype=np.float32), np.array(data_std, dtype=np.float32))
    boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(clip, 1, torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes

