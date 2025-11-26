import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resnet18(class_names):
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('resnet18_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_vgg16(class_names):
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier[6] = nn.Linear(4096, len(class_names))
    model.load_state_dict(torch.load('vgg16_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities.squeeze().cpu().numpy()

def ensemble_voting(pred_resnet, pred_vgg):
    preds = [pred_resnet, pred_vgg]
    return max(set(preds), key=preds.count)

def ensemble_averaging(prob_resnet, prob_vgg):
    avg_prob = (prob_resnet + prob_vgg) / 2
    return np.argmax(avg_prob)

def ensemble_weighted(prob_resnet, prob_vgg, weight_resnet):
    weight_vgg = 1.0 - weight_resnet
    avg_prob = (weight_resnet * prob_resnet + weight_vgg * prob_vgg)
    return np.argmax(avg_prob), avg_prob