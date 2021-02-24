from codes.Mydatasets.Classification import classification_dataset, classification_predict_dataset
from codes.Mydatasets.LishuiYaogan import Yaogan_dataset, Yaogan_dataset_predict_dataset

dataset_dict = {
    'Yaogan_dataset': Yaogan_dataset,
    'Yaogan_dataset_predict_dataset': Yaogan_dataset_predict_dataset,
}


def dataset_call(flag, kwargs):
    dataset_name = kwargs['dataset_name']
    assert dataset_name in dataset_dict.keys()
    dataset = dataset_dict[dataset_name](flag, kwargs)
    return dataset
