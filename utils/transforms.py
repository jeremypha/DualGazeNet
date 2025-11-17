
import torch
from torchvision import transforms
from torchvision.transforms.functional import normalize
import random


class Resize(object):
    def __init__(self, input_size=512):
        self.input_transform = transforms.Resize((input_size, input_size))

    def __call__(self, sample):
        sample["image"] = self.input_transform(sample["image"].unsqueeze(0)).squeeze()
        sample["gt"] = self.input_transform(sample["gt"].unsqueeze(0)).squeeze()

        return sample


class Normalize(object):
    def __init__(self, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, sample):
        sample['image'] = normalize(sample['image'], self.mean, self.std)

        return sample


class RandomHVFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            if random.random() < 0.5:    # H-wise
                if random.random() >= self.prob:
                    sample["image"] = torch.flip(sample["image"], dims=[2])
                    sample["gt"] = torch.flip(sample["gt"], dims=[1])

                else:                   # V-wise
                    sample["image"] = torch.flip(sample["image"], dims=[1])
                    sample["gt"] = torch.flip(sample["gt"], dims=[0])

        return sample


class RandomRotate(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            if random.random() < 0.5:  # Clockwise
                sample["image"] = torch.rot90(sample["image"], k=1, dims=[1, 2])
                sample["gt"] = torch.rot90(sample["gt"], k=1, dims=[0, 1])

            else:  # Counterclockwise
                sample["image"] = torch.rot90(sample["image"], k=-1, dims=[1, 2])
                sample["gt"] = torch.rot90(sample["gt"], k=-1, dims=[0, 1])

        return sample
