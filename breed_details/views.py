from django.http import JsonResponse
from breed_details.models import DogBreedDetails, DogType
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import requests
# Create your views here.

@csrf_exempt
@api_view(['GET'])
def breed_detail_list(request):
    all_breeds = DogBreedDetails.objects.select_related('type_id').all().values('id','breed','avatar','type_id__type')
    return JsonResponse(list(all_breeds), safe=False)

@csrf_exempt
@api_view(['POST','GET'])
def specific_breed_details(request):
    if request.method=='POST':
        input_string = request.data['index']
        detail = DogBreedDetails.objects.get(pk=input_string)
        return JsonResponse({'Description':detail.description,'Breed':detail.breed,
                             'Character':detail.character,'Height':detail.height,
                             'Wight':detail.weight,'Life':detail.life_expentancy,
                             'Akc':detail.akc_link,'Img':detail.image2,'Img0':detail.image0,
                             'Img1':detail.image1,'avatar':detail.avatar
                             })

@csrf_exempt
@api_view(['GET'])
def breed_list(request):
    breed1 = DogBreedDetails.objects.select_related('type_id').filter(pk=1).values('id','breed','avatar','type_id__type')
    breed2 = DogBreedDetails.objects.select_related('type_id').filter(pk=1).values('id','breed','avatar','type_id__type')
    recommended = list(breed1)+list(breed2)
    return JsonResponse(recommended, safe=False)

@csrf_exempt
@api_view(['GET'])
def breed_list(request):
    all_breeds = DogBreedDetails.objects.select_related('type_id').all().values('id','breed','avatar','type_id__type')
    return JsonResponse(list(all_breeds), safe=False)


@csrf_exempt
@api_view(['GET'])
def breed_recommended(request):
    breed1 = DogBreedDetails.objects.select_related('type_id').filter(pk=137).values('id','breed','avatar','type_id__type')
    breed2 = DogBreedDetails.objects.select_related('type_id').filter(pk=100).values('id','breed','avatar','type_id__type')
    recommended = list(breed1)+list(breed2)
    return JsonResponse(recommended, safe=False)

@csrf_exempt
@api_view(['GET'])
def top_breed(request):
    breed1 = DogBreedDetails.objects.select_related('type_id').filter(pk=153).values('id','breed','avatar','type_id__type')
    breed2 = DogBreedDetails.objects.select_related('type_id').filter(pk=73).values('id','breed','avatar','type_id__type')
    breed3 = DogBreedDetails.objects.select_related('type_id').filter(pk=19).values('id','breed','avatar','type_id__type')
    recommended = list(breed1)+list(breed2)+list(breed3)
    return JsonResponse(recommended, safe=False)

