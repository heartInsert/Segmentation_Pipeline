import torch
import segmentation_models_pytorch as smp


class segmentation_models(torch.nn.Module):
    def __init__(self, kwargs):
        super(segmentation_models, self).__init__()
        self.model = smp.Unet(
            encoder_name=kwargs['encoder_name'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=kwargs['encoder_weights'],  # use `imagenet` pretreined weights for encoder initialization
            in_channels=kwargs['in_channels'],  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=kwargs['classes'],  # model output channels (number of classes in your dataset)
        )

    def forward(self, data):
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        out = self.model(data)
        return out
