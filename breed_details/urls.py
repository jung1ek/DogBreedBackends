from . import views
from django.urls import path

app_name = 'breed_details'

urlpatterns = [
     path('breed_list/', views.breed_detail_list),
     path('breed_detail/', views.specific_breed_details),
     path('test/', views.breed_list),
     path('recommended/', views.breed_recommended),
     path('top/', views.top_breed),]
