from django.contrib import admin
from  breed_details.models import DogOrigin, DogType, DogBreedDetails
# Register your models here.
admin.site.register(DogBreedDetails)
admin.site.register(DogType)
admin.site.register(DogOrigin)