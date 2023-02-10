from django.urls import path 
from . import views 
urlpatterns=[
    path('problem_solution/',views.chatbot)
]