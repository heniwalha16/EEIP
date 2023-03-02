from django.urls import path 
from django.conf.urls.static import static
from django.conf import settings
from . import views 
urlpatterns=[
    path('problem_solution/',views.chatbot_solution),
    path('problem_translation/',views.chatbot_translation),
    path('transcribe-speech/', views.transcribe_speech, name='transcribe-speech'),
    path('extract_text/', views.extract_text_from_image, name='extract_text'),
    path('text_to_speech/', views.text_to_speech, name='text_to_speech'),
    #path('teacherApi/', views.teacherApi, name='teacherApi'),
    path('teacher/',views.teacherApi),
    path('teacher/<str:id>/', views.teacherApi, name='teacher_api'),
    path('Class/',views.ClassApi),
    path('Class/<str:id>/', views.ClassApi, name='Class_api'),
    path('Problem/',views.ProblemApi),
    path('Problem/<str:id>/', views.ProblemApi, name='Problem_api'),
    path('SaveFile/', views.SaveFile),
    path('quiz/', views.quiz_view, name='quiz_view'),
    path('quiz/results/', views.quiz_results, name='quiz_results')
    ]

