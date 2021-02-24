import torch
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, jaccard, LovaszLoss
from pytorch_toolbelt import losses as L

