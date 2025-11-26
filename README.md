# Ensemble Medical Image Classification with Streamlit

This project provides a Streamlit web application for training, evaluating, and deploying ensemble predictions using ResNet-18 and VGG-16 models on medical image datasets (blood cells and chest X-rays).

## Features

- **Dataset Management**: Automatic download and preparation of blood cell and chest X-ray datasets
- **Model Training**: Fine-tune ResNet-18 and VGG-16 models with configurable epochs
- **Ensemble Prediction**: Combine model predictions using voting, averaging, or weighted averaging
- **Model Evaluation**: Evaluate individual models and ensemble on test data
- **Interactive UI**: User-friendly Streamlit interface for all operations

## Datasets

- **Blood Cells Dataset**: Classifies white blood cells into EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL
- **Chest X-ray Pneumonia Dataset**: Classifies chest X-rays into NORMAL and PNEUMONIA

## Models

- **ResNet-18**: Fine-tuned with layer4 and fully connected layer unfrozen
- **VGG-16**: Fine-tuned with the last convolutional block and classifier unfrozen

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision streamlit pillow numpy matplotlib seaborn scikit-learn kagglehub
```

## Usage

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/Coding-with-Akrash/Transfer-learning.git
cd Transfer-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run ensemble_app.py
```

4. Open your browser to `http://localhost:8501`

### Navigation

- **Dataset**: Download and prepare datasets
- **Train Models**: Train ResNet-18 or VGG-16 with selected epochs
- **Ensemble Prediction**: Upload images for ensemble predictions
- **Evaluate Models**: Test accuracy on trained models

## Deployment

### Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select the forked repository
4. Set the main file path to `ensemble_app.py`
5. Click Deploy

### Local Deployment

For production deployment, consider using Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ensemble_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ensemble-app .
docker run -p 8501:8501 ensemble-app
```

## Project Structure

- `ensemble_app.py`: Main Streamlit application
- `train.py`: Model training functions
- `dataset.py`: Dataset download and preparation
- `utils.py`: Utility functions for model loading and prediction
- `main.py`: Original training script (legacy)
- `requirements.txt`: Python dependencies
- `dataset/`: Local dataset storage (created automatically)
- `resnet18_model.pth`: Saved ResNet-18 model
- `vgg16_model.pth`: Saved VGG-16 model

## License

This project is for educational purposes. Datasets are from Kaggle (public domain).

## References

- Blood Cells Dataset: https://www.kaggle.com/paultimothymooney/blood-cells
- Chest X-ray Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- PyTorch Documentation: https://pytorch.org/
- Streamlit Documentation: https://docs.streamlit.io/