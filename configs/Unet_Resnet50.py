import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

Train_mode = True
current_dir = r'/home/xjz/Desktop/Coding/PycharmProjects/competition/tianchi/LishuiYaogan_Sementation'
height, width = 256, 256
kfold = 5
# model setting
model_entity = dict(
    model_name='segmentation_models_pytorch',
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3,
    classes=10,
)

# dataset setting
from torchvision import transforms
import getpass
import cv2

user_name = getpass.getuser()  # 获取当前用户名
if user_name == 'xjz':

# train_transform = A.Compose([
#     # reszie
#     A.Resize(256, 256),
#     #
#     A.OneOf([
#         A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.Transpose(p=0.5)
#     ]),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(),
# ])
    dataset_entity = dict(
        data_csv_path='/home/xjz/Desktop/Coding/DL_Data/LishuiYaogan/train_images_{}_{}'.format(
            height, width),
        data_folder_path='/home/xjz/Desktop/Coding/DL_Data/LishuiYaogan/train_images_{}_{}'.format(
            height, width),
        dataset_name='Yaogan_dataset',
        train_transforms=A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),

        test_transforms=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    )

dataset_entity_predict = dict(
    data_folder_path='/home/xjz/Desktop/Coding/DL_Data/cassava_leaf_disease_classification/test_images',
    dataset_name='Yaogan_dataset_predict_dataset',
    predict_transforms=A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]),
    num_TTA=2,
)
dataloader_entity = dict(
    batch_size=45,
    shuffle=True if Train_mode else False,
    num_wokers=6 if Train_mode else 0,
    drop_last=False,
)
# Trainer setting
trainer_entity = dict(
    gpus=1,
    max_epochs=12 if Train_mode else 4,
    check_val_every_n_epoch=1,
    deterministic=True,
    amp_level='O2',
    precision=16,
)
# loss setting
loss_fc_entity = [
    dict(
        loss_name="DiceLoss",
        loss_args=dict(mode='multiclass'),
        weight=0.5
    ),
    dict(
        loss_name="SoftCrossEntropyLoss",
        loss_args=dict(smooth_factor=0.1),
        weight=0.5
    ),
]
# optimizer setting
optimzier_entity = dict(
    optimizer_name='ranger',
    optimizer_args=dict(
        lr=1e-3,
        use_gc=False,
    )
)
lrschdule_entity = dict(
    lrschdule_name='polynomial_decay_schedule_with_warmup',
    lrschdule_args=dict(
        num_warmup_steps=3 if Train_mode else 2,
        num_training_steps=trainer_entity['max_epochs'],
        lr_end=1e-6,
        power=1.2,
        last_epoch=-1
    ),
    SWA=dict(
        SWA_enable=True,
        SWA_start_epoch=5,
    )
)
training_way = dict(
    training_way_name='Normal_training',
    # optional  Fimix or cutmix
    training_way_args=dict()
)
# logger_setting
logger_entity = dict(
    weight_savepath=os.path.join(current_dir, 'model_weights'),
    ModelCheckpoint=dict(
        monitor='val_Iou_epoch',
        verbose=True,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="max",
        period=1,
        prefix="",
    )

)
