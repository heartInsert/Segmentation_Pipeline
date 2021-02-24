import torch.utils.data as data
import torch
import os
import cv2


class Yaogan_dataset(data.Dataset):
    def __init__(self, flag: str, kwargs):
        assert flag in ['train', 'test']
        self.kwargs = kwargs
        self.data_csv, self.data_transforms = None, None
        if flag == 'train':
            self.flag_train(kwargs)
        else:
            self.flag_test(kwargs)

    def flag_train(self, kwargs):
        self.data_csv = kwargs['train_csv']
        # 注意没有ToTensor
        self.data_transforms = kwargs['train_transforms']

    def flag_test(self, kwargs):
        self.data_csv = kwargs['test_csv']
        self.data_transforms = kwargs['test_transforms']

    def __getitem__(self, item):
        row = self.data_csv.iloc[item]
        file_path = row['file_path']
        label_path = row['label_path']

        data = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        transformed = self.data_transforms(image=data, mask=label)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask'] - 1
        return transformed_image, transformed_mask.long()

    def __len__(self):
        return len(self.data_csv)


class Yaogan_dataset_predict_dataset(data.Dataset):
    def __init__(self, flag: str, kwargs):
        self.kwargs = kwargs
        self.data_csv, self.data_transforms = None, None
        self.flag_predict(kwargs)
        self.data_folder_path = kwargs['data_folder_path']

    def flag_predict(self, kwargs):
        self.data_csv = kwargs['predict_csv']
        self.data_transforms = kwargs['predict_transforms']

    def __getitem__(self, item):
        row = self.data_csv.iloc[item]
        data_path = os.path.join(self.data_folder_path, row['image_id'])
        img = self.data_transforms(data_path)
        return img

    def __len__(self):
        return len(self.data_csv)
