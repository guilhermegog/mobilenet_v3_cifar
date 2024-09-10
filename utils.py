import torch
import torchvision
from torchvision import transforms
from torch import nn
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
import random

from siesta_mobilenet_cifar import mobilenet_v3_large, CosineLinear


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remap_labels(dataset, selected_classes, class_mapping):
    filtered_data = []
    filtered_labels = []

    for img, label in dataset:
        if label in selected_classes:
            # Remap the label
            remapped_label = class_mapping[label]
            filtered_data.append(img)
            filtered_labels.append(remapped_label)

    # Convert to PyTorch tensors
    filtered_data = torch.stack(filtered_data)
    filtered_labels = torch.tensor(filtered_labels)
    return filtered_data, filtered_labels


class CustomCIFAR100(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_cifar100_subset(num_classes=10, batch_size=128):
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5071, 0.4867, 0.4408),
                     (0.2675, 0.2565, 0.2761))
    ])

    full_dataset = torchvision.datasets.CIFAR100(
        root='/space/gguedes/datasets/cifar100/data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root='/space/gguedes/datasets/cifar100/data', train=False, download=True, transform=transform)

    # Randomly select 10 classes
    all_classes = list(range(100))
    selected_classes = random.sample(all_classes, num_classes)

    class_mapping = {original: new for new,
                     original in enumerate(selected_classes)}

    train_data, train_labels = remap_labels(
        full_dataset, selected_classes, class_mapping)
    test_data, test_labels = remap_labels(
        test_dataset, selected_classes, class_mapping)

    train_subset = CustomCIFAR100(train_data, train_labels)
    test_subset = CustomCIFAR100(test_data, test_labels)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader, selected_classes


def adapt_mbnet():
    model = mobilenet_v3_large()

    model.classifier[3] = CosineLinear(model.classifier[3].in_features, 10)
    return model


def load_partial_model(model, checkpoint_path, num_frozen_layers=8):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()

    # Filter out layers after the 8th layer
    pretrained_dict = {k: v for k, v in checkpoint.items(
    ) if k in model_dict and 'features.' in k and int(k.split('.')[1]) < num_frozen_layers}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    # Freeze the first 8 layers
    for name, param in model.named_parameters():
        if 'features.' in name and int(name.split('.')[1]) < num_frozen_layers:
            param.requires_grad = False

    return model


def save_partial_model(model, save_path, start_layer=8):
    state_dict = model.state_dict()
    save_dict = {k: v for k, v in state_dict.items(
    ) if 'features.' not in k or int(k.split('.')[1]) >= start_layer}
    torch.save(save_dict, save_path)
