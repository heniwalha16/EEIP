from django.urls import path 
from django.conf.urls.static import static
from django.conf import settings
from . import views 
urlpatterns=[
    path('intro/',views.intro,name='intro'),

    path('problem_solution/',views.chatbot_solution),
    path('problem_translation/',views.chatbot_translation),
    path('transcribe-speech/', views.transcribe_speech, name='transcribe-speech'),
    path('extract_text/', views.extract_text_from_image, name='extract_text'),
    path('text_to_speech/', views.text_to_speech, name='text_to_speech'),
    #path('teacherApi/', views.teacherApi, name='teacherApi'),
    path('teacher/',views.teacherApi),
    path('teacher/<str:id>/', views.teacherApi, name='student_api'),
    path('student/',views.StudentApi),
    path('student/<str:id>/', views.StudentApi, name='student_api'),
    path('Class/',views.ClassApi),
    path('Class/<str:id>/', views.ClassApi, name='Class_api'),
    path('Problem/',views.ProblemApi),
    path('Problem/<str:id>/', views.ProblemApi, name='Problem_api'),
    path('SaveFile/', views.SaveFile),
#    path('quiz/', views.quiz_view, name='quiz_view'),
    #path('main/', views.home, name='main_view'),
    path('calculate/', views.calculate, name='calculate'),
    path('Tik_Tac_Toe/', views.Tik_Tac_Toe, name='Tik_Tac_Toe'),
    path('MathQuiz/', views.MathQuiz, name='MathQuiz'),
    path('RockPaperScissors/', views.RockPaperScissors, name='RockPaperScissors'),
    path('LogOut/', views.LogOut, name='LogOut'),


    #path('quiz/results/', views.quiz_results, name='quiz_results'),
    path('image_generation/', views.image_generation),
    path('login_employee/', views.login_employee, name='login_employee')

    ]

