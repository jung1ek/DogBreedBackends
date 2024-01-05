from django.shortcuts import render
from torchvision import transforms, models
import torch
import os
import json
from torch import nn
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from transformers import AutoImageProcessor, AutoModelForImageClassification
from recognition.vit_selftrained import PretrainViT
# Create your views here.



# setting up device to run the model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def viTModel():  
  # Creating the model and passing the pretrained weights value.
    model = PretrainViT()
    model.load_state_dict(torch.load("recognition/vit_net.pt", map_location=device))
    model.to(device)
    return model
    
def viTStandfordModel():
    model  = AutoModelForImageClassification.from_pretrained("recognition/vit_120_breeds")
    model.to(device)
    return model
    
def convNEtV2Model():
    
    model = AutoModelForImageClassification.from_pretrained("recognition/convnet_133_breeds")
    model.to(device)
    return model
# image processor for standford vit and conv_net
image_processor_vit = AutoImageProcessor.from_pretrained("recognition/vit_120_breeds")
image_processor_convnet = AutoImageProcessor.from_pretrained("recognition/convnet_133_breeds")
#values for preprocess of image vit model
vit_valid_transform_fn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
])


@csrf_exempt
@api_view(['POST'])
def predict_from_vit(request):
    idx = open('recognition/idx_to_breed.json')
    idx_to_breeds = json.load(idx)
    idx.close()
    if request.method == 'POST':
        # Assuming the input is an image file in the request
        image_file = request.FILES['image']

        # Load and preprocess the image
        image = Image.open(image_file).convert('RGB')
        ViT = viTModel()
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
    else:
        return JsonResponse({'Error':'Errors'})
            
            
            
@csrf_exempt
@api_view(['POST'])
def predict_from_convnet(request):
    if request.method == 'POST':
        # Assuming the input is an image file in the request
        image_file = request.FILES['image']
        # loading model
        ConvNet = convNEtV2Model()
        # Load and preprocess the image
        image = Image.open(image_file).convert('RGB')
            
        input_tensor = image_processor_convnet(image,return_tensors="pt").to(device)
        with torch.no_grad():
           logits = ConvNet(**input_tensor).logits
        probabilities = torch.nn.functional.softmax(logits[0],dim=0)
        # model predicts one of the 120 Stanford dog breeds classes
        predicted_class_idx = logits.argmax(-1).item()
        
        # Get the predicted dog breed and its probability
        # predicted_breed_index = torch.argmax(probabilities).item()
        predicted_breed_probability = probabilities[predicted_class_idx].item()

        # Return the predicted dog breed and probability as JSON
        
        breed = ConvNet.config.id2label[predicted_class_idx]
        accuracy = f'{predicted_breed_probability*100:.2f}'
        json = {"Predicted Breed": breed, "Accuracy":accuracy}
        response = json
        # response = {'Breed':[],'Accuracy':[]} 
            
        # if (float(accuracy)<75.0):
        #     value, index = torch.topk(probabilities, 3)
        #     for i,idx in enumerate(index):
        #         response['Breed'].append(idx_to_breeds[str(idx.item())])
        #         response['Accuracy'].append(str(f'{value[i].item()*100:.2f}')+'%')
        # else:
        #     response['Breed'].append(idx_to_breeds[str(predicted_breed_index)])
        #     response['Accuracy'].append(str(accuracy+'%'))
        return JsonResponse(response)
    else:
        return JsonResponse({'Error':'Errors'})
    
    
    
@csrf_exempt
@api_view(['POST'])
def predict_from_vit_standford(request):
    if request.method == 'POST':
        # Assuming the input is an image file in the request
        image_file = request.FILES['image']

        # Load and preprocess the image
        image = Image.open(image_file).convert('RGB')
        ViTStandford = viTStandfordModel()
        input_tensor = image_processor_vit(image,return_tensors="pt").to(device)
        with torch.no_grad():
           logits = ViTStandford(**input_tensor).logits
        probabilities = torch.nn.functional.softmax(logits[0],dim=0)
        # model predicts one of the 120 Stanford dog breeds classes
        predicted_class_idx = logits.argmax(-1).item()
        
        # Get the predicted dog breed and its probability
        # predicted_breed_index = torch.argmax(probabilities).item()
        predicted_breed_probability = probabilities[predicted_class_idx].item()

        # Return the predicted dog breed and probability as JSON
        
        breed = ViTStandford.config.id2label[predicted_class_idx]
        accuracy = f'{predicted_breed_probability*100:.2f}'
        json = {"Predicted Breed": breed, "Accuracy":accuracy}
        response = json
        # response = {'Breed':[],'Accuracy':[]} 
            
        # if (float(accuracy)<75.0):
        #     value, index = torch.topk(probabilities, 3)
        #     for i,idx in enumerate(index):
        #         response['Breed'].append(idx_to_breeds[str(idx.item())])
        #         response['Accuracy'].append(str(f'{value[i].item()*100:.2f}')+'%')
        # else:
        #     response['Breed'].append(idx_to_breeds[str(predicted_breed_index)])
        #     response['Accuracy'].append(str(accuracy+'%'))
        return JsonResponse(response)
    else:
        return JsonResponse({'Error':'Errors'})
    

@csrf_exempt
@api_view(['POST','GET'])
def predict_from_both(request):
    if request.method == 'POST':
        # Assuming the input is an image file in the request
        image_file = request.FILES['image']
        # loading model
        ConvNet = convNEtV2Model()
        # Load and preprocess the image
        image = Image.open(image_file).convert('RGB')
            
        input_tensor = image_processor_convnet(image,return_tensors="pt").to(device)
        with torch.no_grad():
           logits = ConvNet(**input_tensor).logits
        probabilities = torch.nn.functional.softmax(logits[0],dim=0)
        # model predicts one of the 120 Stanford dog breeds classes
        predicted_class_idx = logits.argmax(-1).item()
        
        # Get the predicted dog breed and its probability
        predicted_breed_probability = probabilities[predicted_class_idx].item()
        response = {'Model':[],'Breed':[],'Accuracy':[]}
        if (predicted_breed_probability<0.5):
            
            response['Model'] = 'Vit'
            ViTStandford = viTStandfordModel()
            logits = vitRunStandford(image)
            probabilities = torch.nn.functional.softmax(logits[0],dim=0)
            # model predicts one of the 120 Stanford dog breeds classes
            predicted_class_idx = logits.argmax(-1).item()
            # Get the predicted dog breed and its probability
            predicted_breed_probability = probabilities[predicted_class_idx].item()
            if (predicted_breed_probability>0.75):
                breed = ViTStandford.config.id2label[predicted_class_idx]
                accuracy = f'{predicted_breed_probability*100:.2f}'
                response['Breed'] = breed
                response['Accuracy'] = accuracy
            else:
                value, index = torch.topk(probabilities, 3)
                for i,idx in enumerate(index):
                    response['Breed'].append(ViTStandford.config.id2label[idx.item()])
                    response['Accuracy'].append(str(f'{value[i].item()*100:.2f}')+'%')

        else:
            response['Model'] = 'Conv'
            if(predicted_breed_probability>0.80):
                breed = ConvNet.config.id2label[predicted_class_idx]
                response['Breed'] = breed
                response['Accuracy'] = predicted_breed_probability
            else:
                value, index = torch.topk(probabilities, 3)
                for i,idx in enumerate(index):
                    response['Breed'].append(ConvNet.config.id2label[idx.item()])
                    response['Accuracy'].append(str(f'{value[i].item()*100:.2f}')+'%')
        return JsonResponse(response)
    elif request.method == 'GET':
        #return JsonResponse([{'Model':"ViT",'Breed':'German','Accuracy':'99%'}],safe=False)
        return JsonResponse({'Model':"ViT",'Breed':'German','Accuracy':'99%'})
        


def vitRunStandford(image):
    ViTStandford = viTStandfordModel()
    input_tensor = image_processor_vit(image,return_tensors="pt").to(device)
    with torch.no_grad():
        logits = ViTStandford(**input_tensor).logits
    return logits