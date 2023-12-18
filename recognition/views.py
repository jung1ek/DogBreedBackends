from django.shortcuts import render
from torchvision import transforms, models
import torch
import os
import json
from torch import nn
from PIL import Image
from django.http import JsonResponse
# Create your views here.

# Vision Transformer Model class
class PretrainViT(nn.Module):
    def __init__(self):
        super(PretrainViT, self).__init__()
        model = models.vit_b_16(weights=None)
        num_classifier_feature = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(num_classifier_feature, 120)
        )
        self.model = model
        for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)
# setting up device to run the model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# values for preprocess of image
channel_mean = torch.Tensor([0.485, 0.456, 0.406])
channel_std = torch.Tensor([0.229, 0.224, 0.225])
vit_valid_transform_fn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=channel_mean, std=channel_std),
])

# Creating the model and passing the pretrained weights value.
ViT = PretrainViT()
ViT.load_state_dict(torch.load("recognition/vit_net.pt", map_location=device))
ViT.to(device)
ViT.eval()

# Image index to corresponding name 
idx = open('recognition/idx_to_breed.json')
idx_to_breeds = json.load(idx)
idx.close()

# method to invoke while api request.
def predict_dog_breed(request):
    path = os.path.join("recognition/0f341494dfeb1318b50d43a4ae74e138.jpg")
    image = Image.open(path).convert('RGB')
    input_tensor = vit_valid_transform_fn(image).to(device).unsqueeze(0)
    with torch.no_grad():
        output = ViT(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0],dim=0)
    # Get the predicted dog breed and its probability|
    predicted_breed_index = torch.argmax(probabilities).item()
    predicted_breed_probability = probabilities[predicted_breed_index].item()
    breed = idx_to_breeds[str(predicted_breed_index)]
    accuracy = f'{predicted_breed_probability*100:.2f}'
    response = {'Breed':[],'Accuracy':[]} 
    
    if (float(accuracy)<75.0):
        value, index = torch.topk(probabilities, 3)
        for i,idx in enumerate(index):
            response['Breed'].append(idx_to_breeds[str(idx.item())])
            response['Accuracy'].append(str(f'{value[i].item()*100:.2f}')+'%')
    else:
        response['Breed'].append(idx_to_breeds[str(predicted_breed_index)])
        response['Accuracy'].append(str(accuracy+'%'))
    return JsonResponse(response)


def predict(request):
    if request.method == 'POST':
        # Assuming the input is an image file in the request
        image_file = request.FILES['image']

        # Load and preprocess the image
        image = Image.open(image_file).convert('RGB')
            
        input_tensor = vit_valid_transform_fn(image).to(device).unsqueeze(0)
        with torch.no_grad():
            output = ViT(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0],dim=0)

            # Get the predicted dog breed and its probability
        predicted_breed_index = torch.argmax(probabilities).item()
        predicted_breed_probability = probabilities[predicted_breed_index].item()

            # Return the predicted dog breed and probability as JSON
        
        breed = idx_to_breeds[str(predicted_breed_index)]
        accuracy = f'{predicted_breed_probability*100:.2f}'
        response = {'Breed':[],'Accuracy':[]} 
            
        if (float(accuracy)<75.0):
            value, index = torch.topk(probabilities, 3)
            for i,idx in enumerate(index):
                response['Breed'].append(idx_to_breeds[str(idx.item())])
                response['Accuracy'].append(str(f'{value[i].item()*100:.2f}')+'%')
        else:
            response['Breed'].append(idx_to_breeds[str(predicted_breed_index)])
            response['Accuracy'].append(str(accuracy+'%'))
        return JsonResponse(response)
            