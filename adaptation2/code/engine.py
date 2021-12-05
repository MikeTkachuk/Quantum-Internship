import os
import yaml
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import albumentations as A
import numpy as np
import pandas as pd

from .dataset import PolyDataset
from .models import *
from .channel_shift import ChannelShift



metrics = {'dice_loss': smp.utils.losses.DiceLoss(), 
           'f_score': smp.utils.metrics.Fscore(threshold=0.0),
           'iou_score': smp.utils.metrics.IoU(threshold=0.0)}

save_model = False

train_add_targets = {'image': 'image',
                     'gt_mask': 'mask',
                     'filter_mask': 'mask'}

train_tfs = A.Compose([A.Transpose(p=0.5),
                       A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.ShiftScaleRotate(p=0.5),
                       A.Rotate(p=0.5),
                       #ChannelShift(channels=4,p=0.9),
                       #A.RandomResizedCrop(256,256,(0.5,1.0),(1.0,1.0))
                       ],
                      additional_targets=train_add_targets)

val_tfs = A.Compose([],
                    additional_targets=train_add_targets)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

def get_loaders(root_path='',
                dataset_name='dataset',
                mini_tile_size=256,
                preproc_func=None,
                return_sets=True,
                skip_val=False,
                sentinel_scale=False,
               ):

    # preprocess = get_preprocessing(config['backbone'])

    train_set = PolyDataset(tiles_dir='train', 
        polygons_path='train',
        root_path=root_path,
        dataset_name=dataset_name,
        mini_tile_size=mini_tile_size,
        preproc_func=preproc_func,
        tensor_type='torch',
        transforms=train_tfs,
        sentinel_scale=sentinel_scale)

    
    val_set = PolyDataset(tiles_dir='valid', 
        polygons_path='valid',
        root_path=root_path,
        dataset_name=dataset_name,
        preproc_func=preproc_func,
        tensor_type='torch',
        transforms=val_tfs,
        sentinel_scale=sentinel_scale)
    
    test_set = PolyDataset(tiles_dir='test', 
        polygons_path='test',
        root_path=root_path,
        dataset_name=dataset_name,
        preproc_func=preproc_func,
        tensor_type='torch',
        dst_crs='EPSG:32635',
        transforms=val_tfs,
        sentinel_scale=sentinel_scale)
    
    if return_sets:
        if not skip_val:
            return train_set, val_set, test_set
        else:
            return dataset, test_set

