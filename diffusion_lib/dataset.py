import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

class Dataset():
    def __init__(self, dataset_name, channels, batch_size):
        self.dataset = load_dataset(dataset_name)
        self.channels = channels
        self.batch_size = batch_size

    def transforms(self, examples):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t*2) - 1)
        ])

        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

    def get_dataloder(self):
        transformed_dataset = self.dataset.with_transform(self.transforms).remove_columns('label')
        dataloader = DataLoader(transformed_dataset['train'], batch_size=self.batch_size, shuffle=True)
        return dataloader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t*2) - 1)
    ])
    train_dataset = datasets.CIFAR10(
        root='./cifar_train',
        train=True,
        download=True,
        transform=transform
    )
    print(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    for i,(x,y) in enumerate(train_loader):
        print(x.shape)

