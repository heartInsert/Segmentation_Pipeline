import torch
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, JaccardLoss, LovaszLoss
