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
from breed_details.models import DogBreedDetails
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

json_load = open('recognition/modelidx_to_dbidx.json')
indexes = json.load(json_load)
json_load.close()

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
@api_view(['POST'])
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
        response = []
        if (predicted_breed_probability<0.60):
            ViTStandford = viTStandfordModel()
            logits = vitRunStandford(image)
            probabilities = torch.nn.functional.softmax(logits[0],dim=0)
            # model predicts one of the 120 Stanford dog breeds classes
            predicted_class_idx = logits.argmax(-1).item()
            # Get the predicted dog breed and its probability
            predicted_breed_probability = probabilities[predicted_class_idx].item()
            if (predicted_breed_probability>0.40):
                result = {}
                accuracy = f'{predicted_breed_probability*100:.2f}'
                result['id'] = indexes["vitidx_to_dbidx"][str(predicted_class_idx)]
                detail = DogBreedDetails.objects.get(id=result['id'])
                result['breed'] = detail.breed
                result['accuracy'] = accuracy
                result['description'] = detail.description
                result['character']=detail.character
                result['height']=detail.height
                result['Weight']=detail.weight
                result['life_expentancy']=detail.life_expentancy
                result['akc_link']=detail.akc_link
                result['image2']=detail.image2
                result['image0']=detail.image0
                result['image1']=detail.image1
                result['avatar'] =detail.avatar
                response.append(result)
            else:
                value, index = torch.topk(probabilities, 3)
                for i,idx in enumerate(index):
                    result = {}
                    result['id'] = indexes["vitidx_to_dbidx"][str(idx.item())]
                    result['breed'] = DogBreedDetails.objects.get(id=result['id']).breed
                    result['accuracy'] = f'{value[i].item()*100:.2f}'
                    result['avatar'] = DogBreedDetails.objects.get(id=result['id']).avatar
                    detail = DogBreedDetails.objects.get(id=result['id'])
                    result['breed'] = detail.breed
                    result['description'] = detail.description
                    result['character']=detail.character
                    result['height']=detail.height
                    result['Weight']=detail.weight
                    result['life_expentancy']=detail.life_expentancy
                    result['akc_link']=detail.akc_link
                    result['image2']=detail.image2
                    result['image0']=detail.image0
                    result['image1']=detail.image1
                    response.append(result)

        else:
            if(predicted_breed_probability>0.75):
                result = {}
                result['id'] = indexes["convnetidx_to_dbidx"][str(predicted_class_idx)]
                result['accuracy'] = f'{predicted_breed_probability*100:.2f}'
                detail = DogBreedDetails.objects.get(id=result['id'])
                result['description'] = detail.description
                result['breed']=detail.breed
                result['character']=detail.character
                result['height']=detail.height
                result['Weight']=detail.weight
                result['life_expentancy']=detail.life_expentancy
                result['akc_link']=detail.akc_link
                result['image2']=detail.image2
                result['image0']=detail.image0
                result['image1']=detail.image1
                result['avatar'] =detail.avatar
                response.append(result)
            else:
                value, index = torch.topk(probabilities, 3)
                for i,idx in enumerate(index):
                    result = {}
                    result['id'] = indexes["convnetidx_to_dbidx"][str(idx.item())]
                    result['breed'] = DogBreedDetails.objects.get(id=result['id']).breed
                    result['accuracy'] = f'{value[i].item()*100:.2f}'
                    result['avatar'] = DogBreedDetails.objects.get(id=result['id']).avatar
                    detail = DogBreedDetails.objects.get(id=result['id'])
                    result['breed'] = detail.breed
                    result['description'] = detail.description
                    result['character']=detail.character
                    result['height']=detail.height
                    result['Weight']=detail.weight
                    result['life_expentancy']=detail.life_expentancy
                    result['akc_link']=detail.akc_link
                    result['image2']=detail.image2
                    result['image0']=detail.image0
                    result['image1']=detail.image1
                    result['avatar'] =detail.avatar
                    response.append(result)
        
        # return JsonResponse(response)
        return JsonResponse(response, safe=False)


def vitRunStandford(image):
    ViTStandford = viTStandfordModel()
    input_tensor = image_processor_vit(image,return_tensors="pt").to(device)
    with torch.no_grad():
        logits = ViTStandford(**input_tensor).logits
    return logits

