import torch
import torchvision.transforms as transforms
import numpy as np

print('our transforms')
def image_transforms(mode='train', augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                     do_augmentation=True, transformations=None, size=(256, 512)):
    if mode == 'train':
        data_transform = transforms.Compose([
            ResizeImage(train=True, size=size),
            RandomFlip(do_augmentation),
            ToTensor(train=True),
            AugmentImagePair(augment_parameters, do_augmentation)
        ])
        return data_transform
    elif mode == 'test':
        data_transform = transforms.Compose([
            ResizeImage(train=False, size=size),
            ToTensor(train=False),
            DoTest(),
        ])
        return data_transform
    elif mode == 'custom':
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        print('Wrong mode')


class ResizeImage(object):
    def __init__(self, train=True, size=(256, 512)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.train:

            out = {'left_image': self.transform(sample['left_image']), 'right_image': self.transform(sample['right_image'])}
        else:
            out = self.transform(sample)
        return out


class DoTest(object):
    def __call__(self, sample):
        new_sample = torch.stack((sample, torch.flip(sample, [2])))
        return new_sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.train:
            out = {'left_image': self.transform(sample['left_image']),
                   'right_image': self.transform(sample['right_image'])}
        else:
            out = self.transform(sample)
        return out


class RandomFlip(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']

        if (np.random.uniform(0, 1, 1) >= 0.5) and (self.do_augmentation):
            sample = {'left_image': self.transform(right_image), 'right_image': self.transform(left_image)}
        else:
            sample = {'left_image': left_image, 'right_image': right_image}

        return sample


class AugmentImagePair(object):
    def __init__(self, augment_parameters, do_augmentation):
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        if (self.do_augmentation) and (np.random.uniform(0, 1, 1) > 0.5):
            # randomly shift gamma
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            # randomly shift brightness
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            # randomly shift gamma, brightness, color
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)

            for i in range(3):
                left_image[i, :, :] = (left_image[i, :, :] ** random_gamma) * random_brightness * random_colors[i]
                right_image[i, :, :] = (right_image[i, :, :] ** random_gamma) * random_brightness * random_colors[i]

            # saturate
            left_image = torch.clamp(left_image, 0, 1)
            right_image = torch.clamp(right_image, 0, 1)

            sample = {'left_image': left_image, 'right_image': right_image}

        else:
            sample = {'left_image': left_image, 'right_image': right_image}

        return sample