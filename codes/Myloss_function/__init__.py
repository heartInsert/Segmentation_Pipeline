from codes.Myloss_function.segematation import DiceLoss, SoftCrossEntropyLoss, JaccardLoss
import torch

loss_dict = {
    "DiceLoss": DiceLoss,
    "SoftCrossEntropyLoss": SoftCrossEntropyLoss,
    "jaccard": JaccardLoss,
}


def init_criterion(kwargs):
    loss_name = kwargs['loss_name']
    loss_args = kwargs['loss_args']
    assert loss_name in loss_dict.keys()
    loss_fc = loss_dict[loss_name](**loss_args)
    return loss_fc


def loss_call(kwargs):
    loss = joint_loss(kwargs)
    return loss


class joint_loss(torch.nn.Module):
    def __init__(self, kwargs):
        super(joint_loss, self).__init__()
        if not isinstance(kwargs, list):
            kwargs = [kwargs]
        self.criterion_list = []
        self.criterion_name_list = []
        self.weight_list = []
        assert len(kwargs) >= 1
        for kwarg in kwargs:
            criterion = init_criterion(kwarg)
            self.criterion_list.append(criterion)
            self.criterion_name_list.append(kwarg['loss_name'])
            self.weight_list.append(kwarg.get('weight', 1))

    def forward(self, pred, labels):
        loss_backward = 0
        dict_loss = {'loss_wise': {}}
        for criterion, criterion_name, weight in zip(self.criterion_list, self.criterion_name_list, self.weight_list):
            criterion_loss = criterion(pred, labels) * weight
            loss_backward = loss_backward + criterion_loss
            dict_loss['loss_wise'][criterion_name] = criterion_loss.item()
        return {"loss_backward": loss_backward, **dict_loss}
