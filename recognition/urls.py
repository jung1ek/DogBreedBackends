from . import views
from django.urls import path

app_name = 'recognition'
urlpatterns = [
     path('predict_from_vit/', views.predict_from_vit_standford),
     path('predict_from_convnet/', views.predict_from_convnet),
]
