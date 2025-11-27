from train import train_vgg16

if __name__ == "__main__":
    dataset_path = "dataset/TRAIN"
    train_losses, test_losses, train_accs, test_accs = train_vgg16(dataset_path, num_epochs=5)
    print(f"Final VGG-16 Test Accuracy: {test_accs[-1]:.2f}%")