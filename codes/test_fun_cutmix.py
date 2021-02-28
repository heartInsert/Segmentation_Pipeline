import torch
from mmcv import Config
from pytorch_lightning import loggers
import datetime, shutil, os
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from codes.Mymodels import model_call
from codes.Mydatasets import dataset_call
from codes.Myoptimizers import optimizer_call

from codes.Mylr_schedule import lrschdule_call
from pytorch_lightning.metrics.metric import Metric
from torch.optim.swa_utils import AveragedModel
from codes.Utils.Bn_updater import update_bn
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from segmentation_models_pytorch.losses._functional import soft_jaccard_score, to_tensor


class IOU_loss(_Loss):

    def __init__(
            self,
            mode: str,
            classes: Optional[List[int]] = None,
            log_loss: bool = False,
            from_logits: bool = True,
            smooth: float = 0.,
            eps: float = 1e-7,
    ):
        """Implementation of Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(IOU_loss, self).__init__()

        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            y_pred = y_pred.argmax(1)
            y_pred = F.one_hot(y_pred, num_classes)  # N,H*W -> N,H*W, C
            y_pred = y_pred.permute(0, 2, 1)  # H, C, H*W

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)

        # if self.log_loss:
        #     loss = -torch.log(scores.clamp_min(self.eps))
        # else:
        #     loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        scores *= mask.float()

        if self.classes is not None:
            scores = scores[self.classes]

        return scores.mean()


class Loss_Metric(Metric):
    def __init__(self, compute_on_step: bool = True, dist_sync_on_step=False, process_group=None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)

        self.add_state("loss", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor, num: int):
        self.loss += loss * num
        self.total += num

    def compute(self):
        return self.loss.float() / self.total


class Iou_Metric(Metric):
    def __init__(self, compute_on_step: bool = True, dist_sync_on_step=False, process_group=None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)
        self.Iou_loss = IOU_loss(mode='multiclass')
        self.add_state("Iou_Metric", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(1e-4), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        num = 10
        loss = self.Iou_loss(preds, target)
        self.Iou_Metric += loss.item() * num
        self.total += num

    def compute(self):
        Iou_Metric = self.Iou_Metric / self.total
        return Iou_Metric


from codes.Myloss_function import loss_call

from codes.Mymixway import FMix, Cutmix, training_call


class LitMNIST(LightningModule):
    def __init__(self, kwargs):
        super().__init__()
        self.dataset_entity = kwargs['dataset_entity']
        self.dataloader_entity = kwargs['dataloader_entity']
        self.optimizer_entity = kwargs['optimzier_entity']
        self.lrschdule_entity = kwargs['lrschdule_entity']
        self.loss_fc = loss_call(kwargs['loss_fc_entity'])
        self.model_layer = model_call(kwargs['model_entity'])
        self.train_loss = Loss_Metric(compute_on_step=False)
        self.val_loss = Loss_Metric(compute_on_step=False)
        self.wise_loss = {loss_fc['loss_name']: Loss_Metric(compute_on_step=False) for loss_fc in
                          kwargs['loss_fc_entity']}
        self.train_Accuracy = Iou_Metric(compute_on_step=False)
        self.val_Accuracy = Iou_Metric(compute_on_step=False)
        kwargs['training_way']['training_way_args']['loss_fc'] = self.loss_fc
        self.training_way = training_call(kwargs['training_way'])
        # SWA
        self.SWA_enable = kwargs['lrschdule_entity']['SWA']['SWA_enable']
        self.SWA_start_epoch = kwargs['lrschdule_entity']['SWA']['SWA_start_epoch']
        self.epoch_count = 0
        if self.SWA_enable:
            self.SWA_model = AveragedModel(self.model_layer)

    def train_dataloader(self):
        dataset_train = dataset_call(flag='train', kwargs=self.dataset_entity)

        return DataLoader(dataset_train, batch_size=self.dataloader_entity['batch_size'],
                          shuffle=self.dataloader_entity['shuffle'],
                          num_workers=self.dataloader_entity['num_wokers'],
                          drop_last=self.dataloader_entity['drop_last']
                          )

    def val_dataloader(self):
        dataset_val = dataset_call(flag='test', kwargs=self.dataset_entity)
        return DataLoader(dataset_val, batch_size=self.dataloader_entity['batch_size'],
                          shuffle=False, num_workers=self.dataloader_entity['num_wokers'],
                          drop_last=False
                          )

    def configure_optimizers(self):
        optimizer = optimizer_call(params=self.parameters(), kwargs=self.optimizer_entity)

        lr_schedule = lrschdule_call(optimizer, self.lrschdule_entity)

        return [optimizer], [lr_schedule]

    def forward(self, data):
        x = self.model_layer(data)
        return x

    def step(self, batch):
        data, label = batch[:-1], batch[-1]
        if len(data) == 1:
            data = data[0]
        logits, loss = self.training_way(self, data, label, self.training)
        return loss, logits, label

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        loss_backward = loss['loss_backward']
        # add log
        self.log('train_loss_step', loss_backward, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.train_loss(loss_backward.detach(), len(y))
        self.train_Accuracy(logits.detach(), y)
        return loss_backward

    def training_epoch_end(self, outputs):
        self.log('train_loss_epoch', self.train_loss.compute())
        self.log('train_Iou_epoch', self.train_Accuracy.compute())
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.train_loss.reset()
        self.train_Accuracy.reset()
        self.epoch_count += 1
        if self.SWA_enable and self.epoch_count >= self.SWA_start_epoch:
            self.SWA_model.update_parameters(self.model_layer)

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        # add log
        loss_backward = loss['loss_backward']
        self.val_loss(loss_backward.detach(), len(y))
        self.val_Accuracy(logits.detach(), y)
        for key in loss['loss_wise']:
            self.wise_loss[key](loss['loss_wise'][key], len(y))
            pass

        return loss_backward

    def validation_epoch_end(self, outputs):
        self.log('val_loss_epoch', self.val_loss.compute())
        self.log('val_Iou_epoch', self.val_Accuracy.compute())
        for key in self.wise_loss:
            self.log('{}_epoch'.format(key), self.wise_loss[key].compute())
        self.val_loss.reset()
        self.val_Accuracy.reset()
        for key in self.wise_loss:
            self.wise_loss[key].reset()


from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
import copy
import glob
import numpy as np


def Get_fold_splits(fold, kwargs):
    generate = glob.glob(os.path.join(kwargs['data_csv_path'], '*.tif'))
    X = np.ones([len(generate)])
    skl = KFold(n_splits=fold, shuffle=True, random_state=2020)
    # for train, test in skl.split(X,y):
    #     print('Train: %s | test: %s' % (train, test),'\n')

    index_generator = skl.split(X)
    file_list = [item for item in list(generate)]
    label_list = [item.replace('tif', 'png') for item in file_list]
    csv = pd.DataFrame(np.stack([file_list, label_list]).T, columns=['file_path', 'label_path'])
    return csv, index_generator


def inferrence(model, dataloader):
    pred = []
    target = []
    model.cuda()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data, label = batch[:-1], batch[-1]
            data = [d.cuda() for d in data]
            logits = model(data)
            pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())
            target.extend(label.numpy().tolist())
    del model, dataloader
    return pred, target


def plot_confusion_matrix(pred, target, normalize, save_path):
    if plt.isinteractive():
        plt.ioff()
    fig, axes = plt.subplots(1, 1, figsize=None)
    plot = skplt.metrics.plot_confusion_matrix(target, pred, normalize=normalize, ax=axes)
    fig.savefig(save_path, dpi=300)


if __name__ == "__main__":

    config_dir = r'/home/xjz/Desktop/Coding/PycharmProjects/competition/tianchi/LishuiYaogan_Sementation/configs'
    cofig_name = 'Unet_Resnet50.py'
    # cofig_name = 'debug.py'
    config_path = os.path.join(config_dir, cofig_name)
    seed_everything(2020)
    # get config
    cfg = Config.fromfile(config_path)
    # get  stuff
    model_name = cfg.model_entity['model_name']
    datetime_prefix = datetime.datetime.now().strftime("%Y_%m%d_%H_%M%S")
    experiment_name = datetime_prefix + '_' + model_name
    weight_folder = os.path.join(cfg.logger_entity['weight_savepath'], experiment_name)
    #  copy  config
    os.makedirs(weight_folder)
    shutil.copy(config_path, os.path.join(weight_folder, os.path.basename(config_path)))
    # get dataset length
    csv, index_generator = Get_fold_splits(cfg.kfold, cfg.dataset_entity)
    # pred_list, target_list = [], []
    # for  loop  code
    for n_fold, (train_index, test_index) in enumerate(index_generator, start=1):
        #  testing
        if n_fold <= 1:
            if cfg.Train_mode is False:
                train_index = train_index[:200]
                test_index = test_index[:200]
            cfg['dataset_entity']['train_csv'] = csv.iloc[train_index]
            cfg['dataset_entity']['test_csv'] = csv.iloc[test_index]

            model = LitMNIST(cfg)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(**cfg.logger_entity.ModelCheckpoint)
            tb_logger = loggers.TensorBoardLogger(save_dir=cfg.logger_entity['weight_savepath'], name=experiment_name,
                                                  version='fold_{}'.format(n_fold))
            cfg.trainer_entity['checkpoint_callback'] = checkpoint_callback
            cfg.trainer_entity['logger'] = tb_logger
            trainer = Trainer(**cfg.trainer_entity)
            trainer.fit(model)
            if cfg.Train_mode is not None:
                model_pth_name = os.path.basename(checkpoint_callback.best_model_path)
                best_model_path = checkpoint_callback.best_model_path.replace(model_pth_name, 'best.ckpt')
                if model.SWA_enable:
                    print('SWA save')
                    update_bn(trainer.train_dataloader, model.SWA_model)
                    model.model_layer.load_state_dict(model.SWA_model.module.state_dict())
                else:
                    model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'], strict=True)
                torch.save({"state_dict": model.model_layer.state_dict()}, best_model_path)

                # delete original
                os.remove(checkpoint_callback.best_model_path)
                # confusion matrix
                pred, target = inferrence(model.model_layer, copy.deepcopy(trainer.val_dataloaders[0]))

                # plot_confusion_matrix(pred, target, normalize=False,
                #                       save_path=os.path.join(os.path.dirname(checkpoint_callback.dirpath),
                #                                              'confusion_matrix.jpg'))
                # plot_confusion_matrix(pred, target, normalize=True,
                #                       save_path=os.path.join(os.path.dirname(checkpoint_callback.dirpath),
                #                                              'confusion_matrix_normalize.jpg'))
                # add pred and target to global
                # pred_list.extend(pred)
                # target_list.extend(target)

            del model, trainer, checkpoint_callback
    # plot_confusion_matrix(pred_list, target_list, normalize=False,
    #                       save_path=os.path.join(weight_folder, 'global_confusion_matrix.jpg'))
    # plot_confusion_matrix(pred_list, target_list, normalize=True,
    #                       save_path=os.path.join(weight_folder, 'global_confusion_matrix_normalize.jpg'))
