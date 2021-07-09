import pytorchvideo.models.resnet as pvrn
import torch.nn as nn

def get_resnet_model():
    return pvrn.create_resnet(
        input_channel=3, # RGB input from dataset
        model_depth=50, # depth of ResNet of model
        model_num_class=5, # number of classes for final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )