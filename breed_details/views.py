from django.http import JsonResponse
from breed_details.models import DogBreedDetails, DogType
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import requests
# Create your views here.


@csrf_exempt
@api_view(['GET'])
def breeds_list(request):
    all_breeds = DogBreedDetails.objects.select_related('type_id').all().values('id','breed','avatar','type_id__type')
    return JsonResponse(list(all_breeds), safe=False)

@csrf_exempt
@api_view(['POST','GET'])
def specific_breed_details(request):
    if request.method=='POST':
        input_string = request.data['index']
        return JsonResponse({'Success': input_string})



@csrf_exempt
@api_view(['GET'])
def breeds_list(request):
    
    return JsonResponse(list(), safe=False)