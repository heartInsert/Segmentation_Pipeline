import os, datetime
from mmcv import Config
from codes.Mydatasets import dataset_call
from codes.Mymodels import model_call
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
import cv2


def Get_predict_dataloader(cfg):
    dataset_val = dataset_call(flag='predict', kwargs=cfg.dataset_entity_predict)
    return DataLoader(dataset_val, batch_size=50, shuffle=False, num_workers=6, drop_last=False)


def prediction(model_list, dataloader, predict_way) -> np.array:
    with torch.no_grad():
        for model in model_list:
            model.cuda()
            model.eval()
        result = []
        for index, data in enumerate(dataloader):
            img = data['img'].cuda()
            file_paths = data['file_path']
            for model in model_list:
                pred = model(img)
                if predict_way == 'cross_entropy':
                    pred = pred.log_softmax(dim=1).exp()
                if predict_way == 'sigmoid':
                    pred = pred.log_sigmoid().exp()
                pred = pred.detach().cpu().numpy()
                result.append(pred)
            del pred
            save_result(result, file_paths)

            pass
    return None


def save_result(result, file_paths):
    result = np.stack(result)
    result = result.mean(axis=0)
    result = result.argmax(axis=1)
    result = result + 1
    for index in range(len(result)):
        img = result[0]
        cv2.imwrite("img_dir.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    pass


def make_prediction(model_folder_path, model, dataloader, predict_way):
    model_predict = []
    for fold_n in os.listdir(model_folder_path):
        if fold_n.startswith('fold_'):
            fold_n_path = os.path.join(model_folder_path, fold_n)
            checkpoint_path = os.path.join(fold_n_path, 'checkpoints', 'best.ckpt')
            model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=True)
            result = prediction(model, dataloader, predict_way)
            model_predict.append(result)
            del result
            gc.collect()
    model_predict = np.stack(model_predict)
    return model_predict


import gc


def main():
    project_path = '/home/xjz/Desktop/Coding/PycharmProjects/competition/tianchi/LishuiYaogan_Sementation'
    model_folders = ['2021_0224_20_2441_segmentation_models_pytorch']

    model_folder_paths = [os.path.join(project_path, 'model_weights', folder) for folder in model_folders]
    config_paths = []
    for model_folder_path in model_folder_paths:
        files = os.listdir(model_folder_path)
        for file in files:
            if file.endswith('.py'):
                config_paths.append(os.path.join(model_folder_path, file))
    predict_way = 'cross_entropy'
    assert predict_way in ['cross_entropy', 'sigmoid']
    assert len(config_paths) == len(model_folders)
    final_predicts = []
    generate = glob.glob(
        os.path.join('/home/xjz/Desktop/Coding/DL_Data/LishuiYaogan/suichang_round1_test_partA_210120', '*.tif'))
    file_list = [item for item in list(generate)]
    csv = pd.DataFrame(file_list, columns=['file_path'])
    cfg = Config.fromfile(config_paths[0])
    cfg['dataset_entity_predict']['predict_csv'] = csv
    dataloader = Get_predict_dataloader(cfg)
    model_list = []
    for config_path in config_paths:
        cfg = Config.fromfile(config_path)
        model_list.append(model_call(cfg['model_entity']))
    prediction(model_list, dataloader, predict_way)

    print('Done')


if __name__ == "__main__":
    main()
