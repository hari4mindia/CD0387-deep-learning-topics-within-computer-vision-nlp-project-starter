import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import io


def model_fn(model_dir):
    num_classes = 133
    model = models.resnet50(pretrained=False)  
    for param in model.parameters():
        param.requires_grad = False

    num_inputs = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_inputs, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, num_classes))
    
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def input_fn(request_body, content_type):
    if content_type == "image/jpeg":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)  # add batch dim
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    model.eval()
    with torch.no_grad():
        output = model(input_data)
        return output

def output_fn(prediction, accept):
    if accept == "application/json":
        probs = torch.nn.functional.softmax(prediction, dim=1)
        top_prob, top_class = torch.max(probs, 1)
        return {
            "predicted_class": top_class.item(),
            "probability": top_prob.item()
        }
    else:
        raise ValueError(f"Unsupported accept type: {accept}")