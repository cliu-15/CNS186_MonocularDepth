import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset

print('our utils')

from model import Resnet50
#from models_resnet_original import ResnetModel, Resnet50_md
from data_loader import KittiLoader
from transforms import image_transforms

def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, input_channels=3, pretrained=False):
    model_get = {
        'resnet50_md': Resnet50(input_channels),
        #'resnet18_md': Resnet18_md(input_channels)
    }

    return model_get.get(model, ResnetModel(input_channels, encoder = model, pretrained = pretrained))


def prepare_dataloader(data_directory, mode, augment_parameters,
                       do_augmentation, batch_size, size, num_workers):
    data_dirs = os.listdir(data_directory)
    data_transform = image_transforms(mode = mode, augment_parameters = augment_parameters, do_augmentation = do_augmentation, size = size)
    datasets = [KittiLoader(os.path.join(data_directory, data_dir), mode, transform=data_transform)
                            for data_dir in data_dirs]
    dataset = ConcatDataset(datasets)
    n_img = len(dataset)
    print('Use a dataset with', n_img, 'images')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle = (mode == 'train'), num_workers=num_workers, pin_memory=True)
    return n_img, loader