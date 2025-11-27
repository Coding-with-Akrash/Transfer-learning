import torch
import matplotlib.pyplot as plt
import os
from train import load_dataset
from utils import load_resnet18, load_vgg16, load_densenet121, load_efficientnet_b0, load_mobilenet_v2, load_vit_b_16, load_cnn_lstm

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    dataset_path = "dataset/TRAIN"
    train_loader, test_loader, class_names = load_dataset(dataset_path)

    accuracies = {}
    models_to_evaluate = []

    if os.path.exists('resnet18_model.pth'):
        print("Evaluating ResNet-18...")
        resnet_model = load_resnet18(class_names)
        resnet_acc = evaluate_model(resnet_model, test_loader)
        accuracies['ResNet-18'] = resnet_acc
        models_to_evaluate.append('ResNet-18')
        print(f"ResNet-18 Test Accuracy: {resnet_acc:.2f}%")

    if os.path.exists('vgg16_model.pth'):
        print("Evaluating VGG-16...")
        vgg_model = load_vgg16(class_names)
        vgg_acc = evaluate_model(vgg_model, test_loader)
        accuracies['VGG-16'] = vgg_acc
        models_to_evaluate.append('VGG-16')
        print(f"VGG-16 Test Accuracy: {vgg_acc:.2f}%")

    if os.path.exists('densenet121_model.pth'):
        print("Evaluating DenseNet-121...")
        densenet_model = load_densenet121(class_names)
        densenet_acc = evaluate_model(densenet_model, test_loader)
        accuracies['DenseNet-121'] = densenet_acc
        models_to_evaluate.append('DenseNet-121')
        print(f"DenseNet-121 Test Accuracy: {densenet_acc:.2f}%")

    if os.path.exists('efficientnet_b0_model.pth'):
        print("Evaluating EfficientNet-B0...")
        efficientnet_model = load_efficientnet_b0(class_names)
        efficientnet_acc = evaluate_model(efficientnet_model, test_loader)
        accuracies['EfficientNet-B0'] = efficientnet_acc
        models_to_evaluate.append('EfficientNet-B0')
        print(f"EfficientNet-B0 Test Accuracy: {efficientnet_acc:.2f}%")

    if os.path.exists('mobilenet_v2_model.pth'):
        print("Evaluating MobileNetV2...")
        mobilenet_model = load_mobilenet_v2(class_names)
        mobilenet_acc = evaluate_model(mobilenet_model, test_loader)
        accuracies['MobileNetV2'] = mobilenet_acc
        models_to_evaluate.append('MobileNetV2')
        print(f"MobileNetV2 Test Accuracy: {mobilenet_acc:.2f}%")

    if os.path.exists('vit_b_16_model.pth'):
        print("Evaluating ViT-B/16...")
        vit_model = load_vit_b_16(class_names)
        vit_acc = evaluate_model(vit_model, test_loader)
        accuracies['ViT-B/16'] = vit_acc
        models_to_evaluate.append('ViT-B/16')
        print(f"ViT-B/16 Test Accuracy: {vit_acc:.2f}%")

    if os.path.exists('cnn_lstm_model.pth'):
        print("Evaluating CNN-LSTM...")
        cnn_lstm_model = load_cnn_lstm(class_names)
        cnn_lstm_acc = evaluate_model(cnn_lstm_model, test_loader)
        accuracies['CNN-LSTM'] = cnn_lstm_acc
        models_to_evaluate.append('CNN-LSTM')
        print(f"CNN-LSTM Test Accuracy: {cnn_lstm_acc:.2f}%")

    if not accuracies:
        print("No trained models found. Please train the models first.")
        return

    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green'])
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 100)
    for i, (model, acc) in enumerate(accuracies.items()):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')
    plt.savefig('model_accuracy_comparison.png')
    print("Accuracy comparison graph saved as 'model_accuracy_comparison.png'")

if __name__ == "__main__":
    main()