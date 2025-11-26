import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import kagglehub
import shutil
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if dataset is already downloaded locally
local_dataset_path = 'dataset'
train_path = os.path.join(local_dataset_path, 'TRAIN')
if not os.path.exists(train_path):
    print("Downloading blood cells dataset...")
    path_blood = kagglehub.dataset_download("paultimothymooney/blood-cells")
    print("Path to blood cells dataset files:", path_blood)
    # Copy the TRAIN folder to local dataset
    source_train_blood = os.path.join(path_blood, 'dataset2-master', 'dataset2-master', 'images', 'TRAIN')
    shutil.copytree(source_train_blood, train_path)
    print("Blood cells dataset copied to local directory.")

    print("Downloading chest X-ray dataset...")
    path_xray = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to chest X-ray dataset files:", path_xray)
    # Copy the TRAIN folder contents to local dataset TRAIN
    source_train_xray = os.path.join(path_xray, 'chest_xray', 'chest_xray', 'train')
    for item in os.listdir(source_train_xray):
        s = os.path.join(source_train_xray, item)
        d = os.path.join(train_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
    print("Chest X-ray dataset added to local directory.")
else:
    print("Dataset already exists locally.")

dataset_path = train_path

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split into train and test (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(full_dataset.classes)

# Function to train and evaluate model
def train_model(model, train_loader, test_loader, num_epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return train_losses, test_losses, train_accs, test_accs

# Function to predict on a single image
def predict_image(model_path, model_class, image_path, class_names):
    # Load model
    model = model_class(weights='IMAGENET1K_V1')
    if model_class == models.resnet18:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    elif model_class == models.vgg16:
        model.classifier[6] = nn.Linear(4096, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# ResNet-18 Fine-Tuning
print("Training ResNet-18...")
resnet18 = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_classes)

# Freeze all layers
for param in resnet18.parameters():
    param.requires_grad = False

# Unfreeze layer4 and fc
for name, param in resnet18.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

resnet18 = resnet18.to(device)
resnet_train_losses, resnet_test_losses, resnet_train_accs, resnet_test_accs = train_model(resnet18, train_loader, test_loader)
torch.save(resnet18.state_dict(), 'resnet18_model.pth')
print("ResNet-18 model saved.")

# VGG-16 Fine-Tuning
print("\nTraining VGG-16...")
vgg16 = models.vgg16(weights='IMAGENET1K_V1')
vgg16.classifier[6] = nn.Linear(4096, num_classes)

# Freeze all layers
for param in vgg16.parameters():
    param.requires_grad = False

# Unfreeze last conv block (features[24:]) + classifier
for name, param in vgg16.features.named_parameters():
    if int(name.split('.')[0]) >= 24:  # conv5 block
        param.requires_grad = True
for param in vgg16.classifier.parameters():
    param.requires_grad = True

vgg16 = vgg16.to(device)
vgg_train_losses, vgg_test_losses, vgg_train_accs, vgg_test_accs = train_model(vgg16, train_loader, test_loader)
torch.save(vgg16.state_dict(), 'vgg16_model.pth')
print("VGG-16 model saved.")

# Plot comparison graphs on single page
plt.figure(figsize=(14, 10))

# Subplot 1: ResNet-18 Loss
plt.subplot(2, 2, 1)
plt.plot(resnet_train_losses, label='Train Loss')
plt.plot(resnet_test_losses, label='Test Loss')
plt.title('ResNet-18 Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Subplot 2: VGG-16 Loss
plt.subplot(2, 2, 2)
plt.plot(vgg_train_losses, label='Train Loss')
plt.plot(vgg_test_losses, label='Test Loss')
plt.title('VGG-16 Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Subplot 3: ResNet-18 Accuracy
plt.subplot(2, 2, 3)
plt.plot(resnet_train_accs, label='Train Acc')
plt.plot(resnet_test_accs, label='Test Acc')
plt.title('ResNet-18 Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Subplot 4: VGG-16 Accuracy
plt.subplot(2, 2, 4)
plt.plot(vgg_train_accs, label='Train Acc')
plt.plot(vgg_test_accs, label='Test Acc')
plt.title('VGG-16 Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Final ResNet-18 Test Accuracy: {resnet_test_accs[-1]:.2f}%")
print(f"Final VGG-16 Test Accuracy: {vgg_test_accs[-1]:.2f}%")

# Print class names
print("Classes:", full_dataset.classes)

# Example usage for instant prediction (uncomment and provide image_path)
# prediction = predict_image('resnet18_model.pth', models.resnet18, 'path/to/image.jpg', full_dataset.classes)
# print("Predicted class:", prediction)