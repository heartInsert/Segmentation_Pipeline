from codes.Mymodels.segmentation_models import segmentation_models

model_dict = {
    'segmentation_models_pytorch': segmentation_models
}


def model_call(kwargs):
    model_name = kwargs['model_name']
    model = model_dict[model_name](kwargs)
    return model
