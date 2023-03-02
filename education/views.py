from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from education.models import Teacher,Class,Problem
from education.serializers import TeacherSerializer ,ClassSerializer,ProblemSerializer

import random
from education.models import QuizQuestion

from django.core.files.storage import default_storage
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt   
from rest_framework import serializers, viewsets 
from rest_framework.decorators import action 
from rest_framework.response import Response  
from django.views.decorators.csrf import csrf_exempt 
from rest_framework.decorators import api_view, renderer_classes 
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
import openai ,os
from django.http import HttpResponse
import azure.cognitiveservices.speech as speechsdk
from django.shortcuts import render
import pytesseract
from PIL import Image
import pyttsx3
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("OPENAI_KEY",None)
########################### chatbot_solution   ###############################
@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def chatbot_solution(request):
    chatbot_response =None 
    if api_key is not None and  request.method=="POST" :
        openai.api_key=api_key
        user_input =request.POST.get('user_input')
        prompt =user_input
        response =openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=256,
            #stop="."
            temperature=0.5
        )
        print(response)
        chatbot_response=response["choices"][0]["text"]
    return Response(chatbot_response)
########################### chatbot_translation   ###############################

@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def chatbot_translation(request):
    chatbot_response =None 
    if api_key is not None and  request.method=="POST" :
        openai.api_key=api_key
        user_input =request.POST.get('user_input')
        prompt =user_input
        response =openai.Completion.create(
            engine='text-davinci-003',
            prompt='translate this to english : '+prompt,
            max_tokens=256,
            #stop="."
            temperature=0.5
        )
        print(response)
        chatbot_response=response["choices"][0]["text"]
    return Response(chatbot_response)
########################### transcribe_speech   ###############################

@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def transcribe_speech(request):
    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = "eca56e10fd6c41b0b2b47181df088d83", "eastus"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Set the target language to Arabic
    speech_config.speech_recognition_language = "ar-EG"  # or "ar-SA"
    # Creates a recognizer with the given settings
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Say something in Arabic...")
    result = speech_recognizer.recognize_once()
    # Checks result.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return HttpResponse(result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
        return HttpResponse("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            return HttpResponse("Speech Recognition Error")

########################### extract_text_from_image   ###############################

@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def extract_text_from_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        # Open the image using PIL
        img = Image.open(image)
        # Use pytesseract to extract the text
        pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
        text = pytesseract.image_to_string(img, lang='ara')
        # Render the extracted text in a template
        return HttpResponse(text)
    else:
        return HttpResponse("Failed")
########################### text_to_speech   ###############################

@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def text_to_speech(request):
    if request.method == 'POST':
        user_input =request.POST.get('user_input')
        # Initialize the speech engine
        engine = pyttsx3.init()
        ####
        voices = engine.getProperty('voices')
        engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_arSA_NaayfM')
        engine.say(user_input)
        engine.runAndWait()
        return HttpResponse("SUCCESS")
    else:
        return HttpResponse("FAILED")
########################### Quiz   ###############################






# from django.utils import timezone
# from .models import Question
# import random

# def quiz(request):
#     if request.method == "POST":
#         selected_question = Question.objects.filter(operation=request.POST['operation'])
#         selected_question = random.choice(selected_question)
#         if int(request.POST['answer']) == selected_question.answer:
#             message = "Correct! Good job."
#         else:
#             message = f"Incorrect. The correct answer is {selected_question.answer}."
#     else:
#         message = ""
#     return render(request, 'quiz.html', {'message': message})

import random
from .models import QuizQuestion

def quiz_view(request):
    difficulty = request.GET.get('difficulty', 1)
    num_questions = request.GET.get('num_questions', 10)
    questions = []
    for i in range(num_questions):
        operation = random.choice(['+', '-', '*', '/'])
        if operation == '+':
            num1 = random.randint(1, int(difficulty) * 100)
            num2 = random.randint(1, int(difficulty) * 100)
            answer = num1 + num2
        elif operation == '-':
            num1 = random.randint(1, int(difficulty) * 100)
            num2 = random.randint(1, int(difficulty) * 100)
            answer = num1 - num2
        elif operation == '*':
            num1 = random.randint(1, int(difficulty) * 20)
            num2 = random.randint(1, int(difficulty) * 20)
            answer = num1 * num2
        else:
            num2 = random.randint(1, int(difficulty) * 20)
            answer = random.randint(1, int(difficulty) * 20)
            num1 = num2 * answer
        questions.append(QuizQuestion(operation=operation, num1=num1, num2=num2, answer=answer, difficulty=difficulty))
    QuizQuestion.objects.bulk_create(questions)
    quiz_questions = QuizQuestion.objects.filter(difficulty=difficulty).order_by('?')[:num_questions]
    context = {
        'quiz_questions': quiz_questions,
        'difficulty': difficulty,
        'num_questions': num_questions
    }
    print(num_questions)
    print('aaa')
    return render(request, 'quiz.html', context)

def quiz_results(request):
    if request.method == 'POST':
        print('aaaa')
        score = 0
        for key, value in request.POST.items():
            if key.startswith('answer'):
                question_id = int(key.replace('answer', ''))
                question = QuizQuestion.objects.get(id=question_id)
                if int(value) == question.answer:
                    score += 1
                    context = {'score': score,
                            'num_questions': request.POST.get('num_questions'),
                            'difficulty': request.POST.get('difficulty'), }
        return render(request, 'quiz_results.html', context)
    else:
        return redirect('quiz_view')
 

########################### teacherApi   ###############################

@csrf_exempt
def teacherApi(request,id=0):
    if request.method=='GET':
        teachers = Teacher.objects.all()
        teacher_serializer = TeacherSerializer(teachers, many=True)
        return JsonResponse(teacher_serializer.data, safe=False)

    if request.method=='POST':
        teacher_data=JSONParser().parse(request)
        teacher_serializer = TeacherSerializer(data=teacher_data)
        if teacher_serializer.is_valid():
            teacher_serializer.save()
            return JsonResponse("Added Successfully!!" , safe=False)
        return JsonResponse("Failed to Add.",safe=False)
    
    elif request.method=='PUT':
        teacher_data = JSONParser().parse(request)
        teacher=Teacher.objects.get(TeacherId=teacher_data['TeacherId'])
        teacher_serializer=TeacherSerializer(teacher,data=teacher_data)
        if teacher_serializer.is_valid():
            teacher_serializer.save()
            return JsonResponse("Updated Successfully!!", safe=False)
        return JsonResponse("Failed to Update.", safe=False)

    elif request.method=='DELETE':
        teacher=Teacher.objects.get(TeacherId=id)
        teacher.delete()
        return JsonResponse("Deleted Succeffully!!", safe=False)
    ########################### ClassApi   ###############################

@csrf_exempt
def ClassApi(request,id=0):
    if request.method=='GET':
        classes=Class.objects.filter(teacher=id)
        #classes = Class.objects.all()
        classes_serializer = ClassSerializer(classes, many=True)
        return JsonResponse(classes_serializer.data, safe=False)

    elif request.method=='POST':
        class_data=JSONParser().parse(request)
        class_serializer = ClassSerializer(data=class_data)
        if class_serializer.is_valid():
            class_serializer.save()
            return JsonResponse("Added Successfully!!" , safe=False)
        return JsonResponse("Failed to Add.",safe=False)
    
    elif request.method=='PUT':
        class_data = JSONParser().parse(request)
        classes=Class.objects.get(ClassId=class_data['ClassId'])
        class_serializer=ClassSerializer(classes,data=class_data)
        if class_serializer.is_valid():
            class_serializer.save()
            return JsonResponse("Updated Successfully!!", safe=False)
        return JsonResponse("Failed to Update.", safe=False)

    elif request.method=='DELETE':
        classes=Class.objects.get(ClassId=id)
        classes.delete()
        return JsonResponse("Deleted Succeffully!!", safe=False)
        ########################### ProblemApi   ###############################

@csrf_exempt
def ProblemApi(request,id=0):
    if request.method=='GET':
       #problems = Problem.objects.all()
       problems=Problem.objects.filter(Class=id)
       problems_serializer = ProblemSerializer(problems, many=True)
       return JsonResponse(problems_serializer.data, safe=False)

    elif request.method=='POST':
        problem_data=JSONParser().parse(request)
        problem_serializer = ProblemSerializer(data=problem_data)
        if problem_serializer.is_valid():
            problem_serializer.save()
            return JsonResponse("Added Successfully!!" , safe=False)
        return JsonResponse("Failed to Add.",safe=False)
    
    elif request.method=='PUT':
        problem_data = JSONParser().parse(request)
        problems=Problem.objects.get(ProblemId=problem_data['ProblemId'])
        problem_serializer=ProblemSerializer(problems,data=problem_data)
        if problem_serializer.is_valid():
            problem_serializer.save()
            return JsonResponse("Updated Successfully!!", safe=False)
        return JsonResponse("Failed to Update.", safe=False)

    elif request.method=='DELETE':
        problems=Problem.objects.get(ProblemId=id)
        problems.delete()
        return JsonResponse("Deleted Succeffully!!", safe=False)
    
    


    
    
    
@csrf_exempt
def SaveFile(request):
    file=request.FILES['uploadedFile']
    file_name = default_storage.save(file.name,file)

    return JsonResponse(file_name,safe=False)
 
