from django.http import JsonResponse
from breed_details.models import DogBreedDetails, DogType
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
# Create your views here.


@csrf_exempt
@api_view(['GET'])
def breeds_list(request):
    all = DogBreedDetails.objects.all().values()
    type = DogType.objects.filter(pk=all[0]['type_id_id']).values()[0]
    json = {}
    json.update(all[0])
    json.update(type)
    return JsonResponse([json], safe=False)
    