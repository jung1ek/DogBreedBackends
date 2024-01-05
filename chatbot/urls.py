from . import views
from django.urls import path

app_name = 'chatbot'

urlpatterns = [
     path('get_message/', views.retrive),]