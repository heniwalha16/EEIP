from django.urls import path 
from . import views 
urlpatterns=[
    path('problem_solution/',views.chatbot_solution),
    path('problem_translation/',views.chatbot_translation),
    path('transcribe-speech/', views.transcribe_speech, name='transcribe-speech'),
    path('extract_text/', views.extract_text_from_image, name='extract_text'),
    path('text_to_speech/', views.text_to_speech, name='text_to_speech'),
    path('quiz/', views.quiz, name='quiz')
    ]
