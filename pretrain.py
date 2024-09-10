import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import sys
from siesta_mobilenet_cifar import mobilenet_v3_large, CosineLinear

from utils import get_cifar100_subset


def adapt_mbnet():
    model = mobilenet_v3_large()
    model.classifier[3] = CosineLinear(model.classifier[3].in_features, 10)
    return model


def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 10 == 0:  # Print every 10 batches
            tqdm.write(f"Epoch [{epoch+1}] Train - Batch [{i}/{len(train_loader)}] Loss: {
                       running_loss/total:.4f} Acc: {correct/total:.4f}")
            sys.stdout.flush()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
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

        if i % 10 == 0:  # Print every 10 batches
            tqdm.write(f"Epoch [{epoch+1}] Test - Batch [{i}/{len(test_loader)}] Loss: {
                       running_loss/total:.4f} Acc: {correct/total:.4f}")
            sys.stdout.flush()

    test_loss = running_loss / len(test_loader)
    test_acc = correct / total
    return test_loss, test_acc


def get_first_eight_layers(model):
    return nn.Sequential(*list(model.features.children())[:8])


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = adapt_mbnet().to(device)
    train_loader, test_loader, classes = get_cifar100_subset(
        num_classes=10, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    best_acc = 0
    for epoch in range(num_epochs):
        tqdm.write(f"\nEpoch [{epoch+1}/{num_epochs}]")
        sys.stdout.flush()

        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, epoch)

        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        tqdm.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        tqdm.write(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        sys.stdout.flush()

        if test_acc > best_acc:
            best_acc = test_acc
            first_eight_layers = get_first_eight_layers(model)
            torch.save(first_eight_layers.state_dict(),
                       'h_net.pth')
            tqdm.write(f"\nSaved state dictionary of first eight layers. Best accuracy: {
                       best_acc:.4f}")
            sys.stdout.flush()

    print("Trained on:", classes)


if __name__ == "__main__":
    main()
