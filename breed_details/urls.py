from . import views
from django.urls import path

app_name = 'breed_details'

urlpatterns = [
     path('breed_list/', views.breeds_list),]