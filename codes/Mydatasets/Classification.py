import torch.utils.data as data
import torch
import os


class classification_dataset(data.Dataset):
    def __init__(self, flag: str, kwargs):
        assert flag in ['train', 'test']
        self.kwargs = kwargs
        self.data_csv, self.data_transforms = None, None

        if flag == 'train':
            self.flag_train(kwargs)
        else:
            self.flag_test(kwargs)
        self.data_folder_path = kwargs['data_folder_path']

    def flag_train(self, kwargs):
        self.data_csv = kwargs['train_csv']
        # 注意没有ToTensor
        self.data_transforms = kwargs['train_transforms']

    def flag_test(self, kwargs):
        self.data_csv = kwargs['test_csv']
        self.data_transforms = kwargs['test_transforms']

    def __getitem__(self, item):
        row = self.data_csv.iloc[item]
        data_path = os.path.join(self.data_folder_path, row['image_id'])
        label = row['label']
        img = self.data_transforms(data_path)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.data_csv)


class classification_predict_dataset(data.Dataset):
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
