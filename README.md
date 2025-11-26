# Medical Image Classification with Transfer Learning

This project demonstrates transfer learning and fine-tuning of ResNet-18 and VGG-16 models for multi-class medical image classification using blood cell and chest X-ray datasets.

## Datasets

The project combines two medical imaging datasets:
- **Blood Cells Dataset**: Classifies white blood cells into EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL
- **Chest X-ray Pneumonia Dataset**: Classifies chest X-rays into NORMAL and PNEUMONIA

Total classes: 6 (EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL, NORMAL, PNEUMONIA)

## Models

- **ResNet-18**: Fine-tuned with layer4 and fully connected layer unfrozen
- **VGG-16**: Fine-tuned with the last convolutional block and classifier unfrozen

## Features

- Automatic dataset download and preparation
- Model training with Adam optimizer
- Evaluation with confusion matrices
- Model saving for inference
- Instant prediction on new images
- Comparative visualization of training metrics

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Kagglehub

Install dependencies:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn pillow kagglehub
```

## Usage

1. Run the training script:
```bash
python main.py
```

The script will:
- Download and prepare datasets
- Train ResNet-18 and VGG-16 models
- Save trained models as `resnet18_model.pth` and `vgg16_model.pth`
- Display training progress and confusion matrices
- Show comparative loss and accuracy plots

2. For prediction on new images:
Uncomment and modify the prediction example in `main.py`:
```python
prediction = predict_image('resnet18_model.pth', models.resnet18, 'path/to/image.jpg', class_names)
print("Predicted class:", prediction)
```

## Results

The models are evaluated on test accuracy, with comparative plots showing:
- Training and test loss curves
- Training and test accuracy curves

Final accuracies are printed for both models.

## Project Structure

- `main.py`: Main script for training and evaluation
- `resnet.py`: Custom ResNet implementation (not used in main script)
- `dataset/`: Local dataset storage (created automatically)
- `resnet18_model.pth`: Saved ResNet-18 model
- `vgg16_model.pth`: Saved VGG-16 model
- `README.md`: This file

## License

This project is for educational purposes. Datasets are from Kaggle (public domain).

## References

- Blood Cells Dataset: https://www.kaggle.com/paultimothymooney/blood-cells
- Chest X-ray Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- PyTorch Documentation: https://pytorch.org/