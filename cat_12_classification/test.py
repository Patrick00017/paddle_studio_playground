import torch
import pandas as pd
import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
import torch.nn as nn
import torch.nn.functional as F

num_classes = 12


class Cat12TestDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = torchvision.io.read_image(img_path) / 255.0
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


class PretrainedResnet50(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedResnet50, self).__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        self.flatten = nn.Flatten()
        self.classifer = nn.Linear(2048, self.num_classes, bias=True)
        nn.init.xavier_uniform(self.classifer.weight)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.classifer(x)
        return x


simple_net = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.MaxPool2d(kernel_size=3),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=3),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=3),

    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=3),

    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),

    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    nn.Flatten(),
    nn.Linear(32, num_classes)
)


def generate_submissive(weight_path: str):
    dataset_path = 'D:\\code\\python\\datasets'
    cat12_path = os.path.join(dataset_path, 'cat12_classification')
    test_sample_file_path = os.path.join(cat12_path, 'result.csv')
    test_img_path = os.path.join(cat12_path, 'cat_12_test', 'cat_12_test')
    test_transforms = [
        T.Resize([224, 224]),
        T.ToTensor()
    ]
    batch_size = 32
    test_transforms = T.Compose(test_transforms)
    cat_12_dataset = Cat12TestDataset(test_sample_file_path, test_img_path, transform=test_transforms)
    test_loader = DataLoader(cat_12_dataset, batch_size=batch_size, shuffle=False)
    sample_submissive = pd.read_csv(test_sample_file_path, header=None)
    labels = []
    # net = simple_net
    net = PretrainedResnet50(num_classes=num_classes)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    net.eval()
    with torch.no_grad():
        for batch in test_loader:
            imgs, _ = batch
            preds = net(imgs)
            preds = F.softmax(preds, dim=-1)
            label = torch.argmax(preds, dim=-1)
            labels.extend([element.item() for element in label])
    sample_submissive.iloc[:, 1] = labels
    sample_submissive.to_csv('./result.csv', index=False, header=None)


if __name__ == '__main__':
    # weight_path = './cat_12_classification_simple.pth'
    weight_path = './cat_12_classification_resnet50.pth'
    generate_submissive(weight_path)
