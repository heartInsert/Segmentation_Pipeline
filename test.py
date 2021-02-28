import glob
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, jaccard, LovaszLoss
import os
from pytorch_toolbelt import losses as L

model = smp.Unet(
    encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=10,  # model output channels (number of classes in your dataset)
)
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
DiceLoss_fn = DiceLoss(mode='multiclass')
SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                        first_weight=0.5, second_weight=0.5).cuda()
label=cv2.imread('/home/xjz/Desktop/Coding/DL_Data/LishuiYaogan/results/000001.png')

generate = glob.glob(r'/home/xjz/Desktop/Coding/DL_Data/LishuiYaogan/train_images_256_256/*.tif')
for file in generate:
    # IMREAD_ANYCOLOR
    # IMREAD_UNCHANGED
    data = cv2.imread(file.replace('png', 'tif'), cv2.IMREAD_ANYCOLOR)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    label = cv2.imread(file)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    temp = cv2.imread("img_dir.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("img_dir.png", label, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    transformed = transform(image=data, mask=label)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    test = model(transformed_image[None, :])
    pass
print()
