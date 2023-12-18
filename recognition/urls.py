from . import views
from django.urls import path

app_name = 'recognition'
urlpatterns = [
     path('recog/', views.predict_dog_breed)
]
