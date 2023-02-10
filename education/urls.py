from django.urls import path 
from . import views 
urlpatterns=[
    path('problem_solution/',views.chatbot_solution),
    path('problem_translation/',views.chatbot_translation)
    ]
