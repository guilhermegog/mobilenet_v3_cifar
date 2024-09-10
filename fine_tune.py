import torch
from torch import nn
from tqdm import tqdm
import sys
import os
from torchvision.transforms import v2

from utils import (set_seed, get_cifar100_subset,
                   adapt_mbnet, load_partial_model, save_partial_model
                   )

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(f"Number of GPUs visible: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")


def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    cutmix = v2.CutMix(num_classes=10, alpha=1.0)
    mixup = v2.MixUp(num_classes=10, alpha=0.1)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = cutmix_or_mixup(inputs, labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (labels.shape != 2):
            labels, indices = torch.max(labels, dim=1)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        total_correct += predicted.eq(indices).sum().item()
        if i % 10 == 0:
            tqdm.write(f"Epoch [{epoch+1}] Train - Batch [{i}/{len(train_loader)}] Loss: {
                       running_loss/total_samples:.4f} Acc: {total_correct/total_samples:.4f}")
            sys.stdout.flush()

    train_loss = running_loss / len(train_loader)
    train_acc = total_correct / total_samples
    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 10 == 0:
                tqdm.write(f"Epoch [{epoch+1}] Test - Batch [{i}/{len(test_loader)}] Loss: {
                           running_loss/total:.4f} Acc: {correct/total:.4f}")
                sys.stdout.flush()

    test_loss = running_loss / len(test_loader)
    test_acc = correct / total
    return test_loss, test_acc


def create_param_groups(model):
    # Example: Splitting parameters into two groups: base layers (features) and classifier
    # Feature extractor (MobileNet backbone)
    base_params = list(model.features.parameters())
    # Classifier (final layers)
    classifier_params = list(model.classifier.parameters())
    param_groups = []
    current_lr = 0.2
    for i, (name, param) in enumerate(model.named_parameters()):
        if (param.requires_grad == True):
            param_groups.append({
                'params': [param],
                'lr': current_lr
            })
            current_lr *= 0.99

    return param_groups

# Adapt input layer to better suit CIFAR images, a.k.a less downsampling


def change_input(model, new_stride=1, new_padding=1, freeze=False):
    # Store the original weights
    original_weights = model.features[0][0].weight.data.clone()

    # Modify the first convolutional layer
    in_channels = model.features[0][0].in_channels
    out_channels = model.features[0][0].out_channels
    kernel_size = model.features[0][0].kernel_size
    model.features[0][0] = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                           stride=new_stride, padding=new_padding, bias=False)

    # Initialize the new layer with the original weights
    with torch.no_grad():
        model.features[0][0].weight.data[:] = original_weights

    # Modify batch norm if needed
    if new_stride == 1:
        num_features = model.features[0][1].num_features
        model.features[0][1] = torch.nn.BatchNorm2d(num_features, eps=0.001, momentum=0.01,
                                                    affine=True, track_running_stats=True)

    # Freeze the initial layer if requested
    if freeze:
        for param in model.features[0].parameters():
            param.requires_grad = False

    return model


def main():
    set_seed(42)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-100 subset
    train_loader, test_loader, selected_classes = get_cifar100_subset(
        num_classes=10)

    # Create and load the model
    model = adapt_mbnet().to(device)
    model = load_partial_model(
        model, 'best_cosine_softmax_loss_SWAV_sgd_layerlr02_step_MIXUP_CUTMIX_50e_100c.pth', num_frozen_layers=8)

    model = change_input(model).to(device)
    param_groups = create_param_groups(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        param_groups, lr=0.2)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.1)
    num_epochs = 100
    best_acc = 0.0

    # MixUp and CutMix

    for epoch in range(num_epochs):
        tqdm.write(f"\nEpoch [{epoch+1}/{num_epochs}]")
        sys.stdout.flush()

        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, epoch)

        scheduler.step()
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        tqdm.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        tqdm.write(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        sys.stdout.flush()

        # Save the state dictionary if we have a new best accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            save_partial_model(
                model, 'g_net.pth', start_layer=8)
            tqdm.write(f"\nSaved state dictionary from 9th layer up. Best accuracy: {
                       best_acc:.4f}")
            sys.stdout.flush()

    print("Selected CIFAR-100 classes:", selected_classes)


if __name__ == "__main__":
    main()
