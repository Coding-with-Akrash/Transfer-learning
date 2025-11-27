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

def load_densenet121(class_names):
    model = models.densenet121(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('densenet121_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_efficientnet_b0(class_names):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('efficientnet_b0_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_mobilenet_v2(class_names):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('mobilenet_v2_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_vit_b_16(class_names):
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('vit_b_16_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.cnn = models.resnet18(weights='IMAGENET1K_V1')
        self.cnn.fc = nn.Identity()
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.cnn(x)
        features = features.unsqueeze(1)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out

def load_cnn_lstm(class_names):
    model = CNNLSTM(len(class_names))
    model.load_state_dict(torch.load('cnn_lstm_model.pth', map_location=device))
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