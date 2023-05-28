from education.utils import draw_parallelogram, draw_rhombus, drawrect, metric_conversion, draw_circle, replace_numbers_with_digits_ar, replace_numbers_with_digits_en
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from education.models import Student, Teacher,Class,Problem
from education.serializers import StudentSerializer, TeacherSerializer ,ClassSerializer,ProblemSerializer
from django.contrib import messages
from datetime import date


import education.utils
import random
#from education.models import QuizQuestion

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
        user_input = request.data.get('user_input')
        prompt ="give me a hint to help me solve this problem, just a hint don't actually solve it. problem : "+user_input
        response =openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=256,
            #stop="."
            temperature=0.5
        )
        print(response)
        chatbot_response=response["choices"][0]["text"]
    return JsonResponse({'chatbot_response': chatbot_response})
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
        return JsonResponse({'text': result.text})
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
        return JsonResponse({'error': 'No speech could be recognized'})
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            return JsonResponse({'error': 'Speech Recognition Error'})

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
        text = pytesseract.image_to_string(img, lang='eng')
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
'''
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
 
'''
########################### StudentApi   ###############################

@csrf_exempt
def StudentApi(request,id=0):
    if request.method=='GET':
        student = Student.objects.all()
        student_serializer = StudentSerializer(student, many=True)
        return JsonResponse(student_serializer.data, safe=False)

    if request.method=='POST':
        student_data=JSONParser().parse(request)
        student_serializer = StudentSerializer(data=student_data)
        if student_serializer.is_valid():
            student_serializer.save()
            return JsonResponse("Added Successfully!!" , safe=False)
        return JsonResponse("Failed to Add.",safe=False)
    
    elif request.method=='PUT':
        student_data = JSONParser().parse(request)
        student=Student.objects.get(id=student_data['id'])
        student_serializer=StudentSerializer(student,data=student_data)
        if student_serializer.is_valid():
            student_serializer.save()
            return JsonResponse("Updated Successfully!!", safe=False)
        return JsonResponse("Failed to Update.", safe=False)

    elif request.method=='DELETE':
        student=Student.objects.get(id=id)
        student.delete()
        return JsonResponse("Deleted Succeffully!!!", safe=False)
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
        teacher=Teacher.objects.get(id=teacher_data['id'])
        teacher_serializer=TeacherSerializer(teacher,data=teacher_data)
        if teacher_serializer.is_valid():
            teacher_serializer.save()
            return JsonResponse("Updated Successfully!!", safe=False)
        return JsonResponse("Failed to Update.", safe=False)

    elif request.method=='DELETE':
        teacher=Teacher.objects.get(id=id)
        teacher.delete()
        return JsonResponse("Deleted Succeffully!!", safe=False)
################################ ClassApi   #####################################

@csrf_exempt
def ClassApi(request,id=0):
    if request.method=='GET':
        classes = Class.objects.all()
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
        classes=Class.objects.get(id=class_data['id'])
        class_serializer=ClassSerializer(classes,data=class_data)
        if class_serializer.is_valid():
            class_serializer.save()
            return JsonResponse("Updated Successfully!!", safe=False)
        return JsonResponse("Failed to Update.", safe=False)

    elif request.method=='DELETE':
        classes=Class.objects.get(id=id)
        classes.delete()
        return JsonResponse("Deleted Succeffully!!", safe=False)
    
    
        ############################### ProblemApi   ###############################

@csrf_exempt
def ProblemApi(request,id=0):
    if request.method=='GET':
       #problems = Problem.objects.all()
       problems=Problem.objects.filter(Class=id)
       problems_serializer = ProblemSerializer(problems, many=True)
       return JsonResponse(problems_serializer.data, safe=False)

    elif request.method=='POST':
        problem_data=JSONParser().parse(request)
        print(problem_data)
        problem_serializer = ProblemSerializer(data=problem_data)
        print(problem_serializer)
        if problem_serializer.is_valid():
            problem_serializer.save()
            return JsonResponse("Added Successfully!!" , safe=False)
        return JsonResponse("Failed to Add.",safe=False)
    
    elif request.method=='PUT':
        problem_data = JSONParser().parse(request)
        problems=Problem.objects.get(id=problem_data['id'])
        problem_serializer=ProblemSerializer(problems,data=problem_data)
        if problem_serializer.is_valid():
            problem_serializer.save()
            return JsonResponse("Updated Successfully!!", safe=False)
        return JsonResponse("Failed to Update.", safe=False)

    elif request.method=='DELETE':
        problems=Problem.objects.get(id=id)
        problems.delete()
        return JsonResponse("Deleted Succeffully!!", safe=False)
    
    


    
    
    
@csrf_exempt
def SaveFile(request):
    file=request.FILES['uploadedFile']
    file_name = default_storage.save(file.name,file)

    return JsonResponse(file_name,safe=False)
 
################################# Image Generation   ###############################
import torch
import torch.nn as nn
import transformers
import langid
import requests
import stanza
from quantulum3 import parser

class BertForMathProblemClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(BertForMathProblemClassification, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang

action_verbs =  ['Clean', 'Drink', 'Play', 'Turn', 'Sit down', 'Smell', 'Ski', 'Wonder', 'Explain', 'Increase', 'Repeat', 'Bathe', 'Run', 'Tell', 'Hug', 'Sit', 'Plan', 'Wash', 'Start', 'Climb', 'Touch', 'Cook', 'Agree', 'Offer', 'Answer', 'Stand', 'Point', 'Check', 'Receive', 'Collect', 'Stand up', 'Ask', 'Enter', 'Continue', 'Rise', 'Leave', 'Enjoy', 'Dream', 'Paint', 'Shake', 'Learn',  'Carry', 'Follow', 'Speak', 'Write', 'Eat', 'Jump', 'Hold', 'Drive', 'Show', 'Use', 'Finish', 'Move', 'Watch', 'Draw', 'Regard', 'Improve', 'Allow', 'Smile', 'Bow', 'Love', 'Dance', 'Hope', 'Meet', 'Choose', 'Grow', 'Take', 'Walk', 'Open', 'Give', 'Reply', 'Exit', 'Travel', 'Change', 'Think', 'Ride', 'Return', 'Like', 'Close', 'Become', 'Create', 'Send', 'Laugh', 'Cry', 'Hear', 'Help', 'Call', 'Find', 'Save', 'Contribute', 'Prepare', 'Begin', 'Solve', 'Study', 'Join', 'Complete', 'Read', 'Act', 'Catch', 'Hide', 'Sell', 'Talk', 'Want']
action_verbs = [word.lower() for word in action_verbs]


def image_generation(seed):
    #stanza.download('ar')
    
    # This sets up a default neural pipeline in Lang
    print(seed)
    sentences = seed.split('.')
    if (seed.count('.') >= 2) or ((seed.count('.') < 2)and(seed.count('?')>0)):
        deleted_sent = sentences.pop(-1)
        seed = '.'.join(sentences)
    lang=detect_language(seed)
    if lang=='ar':
        seed=replace_numbers_with_digits_ar(seed)
    else:
        seed=replace_numbers_with_digits_en(seed)
    seed1=seed
    print(lang)
    if (lang != 'en'):
        response = requests.get('https://api.mymemory.translated.net/get?q='+seed+'&langpair='+lang+'|en')
        seed1  = response.json()['responseData']['translatedText']
    if ',' in seed:
        seed=seed.replace(',',' , ')
    nlp = stanza.Pipeline(lang, use_gpu=False,
                          processors='tokenize,pos,lemma')
    doc = nlp(seed)
    print(doc)
    res = {'type': None, 'data': []}
    
    # Load the trained model and tokenizer
    model = BertForMathProblemClassification()
    model.load_state_dict(torch.load('education/bert_math_problem_classification.pt'))
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Example math problem
    problem = seed

    # Tokenize the input and convert to tensors
    input_ids = torch.tensor(tokenizer.encode(seed1, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    attention_mask = torch.ones_like(input_ids)

    # Pass the input to the model and get the predicted class
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs).item()

    # Print the predicted class
    class_names = ['Not Geometry', 'Geometry']
    problem_type = class_names[predicted_class]
    
    #problem = seed
    metrics=[] 
    #problem_type='Not Geometry'
    print(problem_type)
    if (problem_type=='Geometry'):      
        language, _ = langid.classify(problem)
        if language != 'en':
            response = requests.get('https://api.mymemory.translated.net/get?q='+problem+'&langpair='+'ar'+'|en')
            translated = response.json()['responseData']['translatedText']
        else:
            translated=problem
        translated_doc=nlp(translated)
        for sent in translated_doc.sentences:
            quants = parser.parse(sent.text)
            for q in quants:
                if q.unit.entity.name == 'length' :
                    print(metrics)
                    metrics.append([float(q.surface.split()[0]), q.unit.uri])
        for i in range(len(metrics)):
            metrics[i]=metric_conversion(metrics[i])    
        print(metrics)        
        if len(metrics) > 0:
            if len(metrics)==1:
                if 'diameter' in problem:
                    #res['type'] = 'diametre'
                    Output_List=[draw_circle((metrics[0][0])/2,str(int(metrics[0][1]))+metrics[0][2])]
                elif 'radius' in problem:
                    #res['type'] = 'radius'
                    Output_List=[draw_circle(metrics[0][0],str(int(metrics[0][1]))+metrics[0][2])]
                else:
                    #res['type'] = 'square'
                    Output_List=[drawrect(metrics[0][0],metrics[0][0],str(int(metrics[0][1]))+metrics[0][2],str(int(metrics[0][1]))+metrics[0][2])]

            elif len(metrics)==2:
                if 'parallelogram' in problem:
                    #res['type'] = 'parallelogram'
                    if(metrics[0][1]==max(metrics[0][1],metrics[1][1])):
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_height=metrics[0][1]
                        r_width=metrics[1][1]
                        c_height=metrics[0][0]
                        c_width=metrics[1][0]
                    else:
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_width=metrics[0][1]
                        r_height=metrics[1][1]
                        c_width=metrics[0][0]
                        c_height=metrics[1][0]
                    Output_List=[draw_parallelogram(c_height,c_width,height,width)]
                elif 'rhombus' in problem:
                    #res['type'] = 'rhombus'
                    if(metrics[0][1]==max(metrics[0][1],metrics[1][1])):
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_height=metrics[0][1]
                        r_width=metrics[1][1]
                        c_height=metrics[0][0]
                        c_width=metrics[1][0]
                    else:
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_width=metrics[0][1]
                        r_height=metrics[1][1]
                        c_width=metrics[0][0]
                        c_height=metrics[1][0]
                    Output_List=[draw_rhombus(c_height,c_width,height,width)]
                else:
                    #res['type'] = 'rectangle'
                    if(metrics[0][1]==max(metrics[0][1],metrics[1][1])):
                        print(str(metrics[0][1]))
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_height=metrics[0][1]
                        r_width=metrics[1][1]
                        c_height=metrics[0][0]
                        c_width=metrics[1][0]
                    else:
                        print(str(int(metrics[0][1])))
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_width=metrics[0][1]
                        r_height=metrics[1][1]
                        c_width=metrics[0][0]
                        c_height=metrics[1][0]
                    print(c_height,c_width,height,width)
                    Output_List=[drawrect(c_height,c_width,height,width)]
            elif len(metrics)==4:
                if 'trapezium' in problem:
                    res['type'] = 'trapezium'
            del doc
            print(len(metrics))
        
            Output_List=[[Output_List[0],3]]
            print(Output_List)
            return Output_List

    else:
        res['type'] = 'entity'
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    for sent in doc.sentences:
        for word in sent.words:
            res['data'].append([word.lemma, word.upos, word.text])

    i = 0
    #for w in res['data']:
     #   w.append(i)
      #  i = 0
       # if w[1] == 'NUM':
        #    print(w[0])
         #   i = int(w[0])
    print(res)
    for i, w in enumerate(res['data']):
        if w[1] == 'NOUN' :
            if i < len(res['data'])-1:
                if res['data'][i+1][1] == 'NOUN':
                    w[0] = w[0]+' '+res['data'][i+1][0]
                    w[2] = w[2]+' '+res['data'][i+1][2]
                    del res['data'][i+1]
    print(res)
    for i, w in enumerate(res['data']):
        if w[1] == 'PROPN':
            if i < len(res['data'])-1:
                if res['data'][i+1][1] == 'PROPN':
                    w[0] = w[0]+' '+res['data'][i+1][0]
                    w[2] = w[2]+' '+res['data'][i+1][2]
                    del res['data'][i+1]
    
    for i, w in enumerate(res['data']):
        if w[1] == 'ADV':
            if (i < len(res['data'])-1) and (i>0):
                print("aaa")
                if (res['data'][i+1][1] == 'VERB') and (res['data'][i+1][0] in action_verbs):
                    w[0] = w[0]+' '+res['data'][i+1][0]
                    w[2] = w[2]+' '+res['data'][i+1][2]
                    del res['data'][i+1]    
                elif (res['data'][i-1][1] == 'VERB') and (res['data'][i-1][0] in action_verbs):
                    w[0] = res['data'][i-1][0] + ' ' + w[0]
                    w[2] = res['data'][i-1][2] + ' ' + w[2]
                    del res['data'][i-1]   

    for i, w in enumerate(res['data']):
        if w[1] == 'X':
            if (i < len(res['data'])-1) and (i>0):
                print("aaa")
                if (res['data'][i+1][1] == 'NOUN') :
                    w[0] = w[0]+' '+res['data'][i+1][0]
                    w[2] = w[2]+' '+res['data'][i+1][2]
                    w[1]='NOUN'
                    del res['data'][i+1]    
                elif (res['data'][i-1][1] == 'NOUN') :
                    w[0] = res['data'][i-1][0] + ' ' + w[0]
                    w[2] = res['data'][i-1][2] + ' ' + w[2]
                    w[1]="NOUN"
                    del res['data'][i-1]   
    print(res)
    dim_numbers=[]
    language, _ = langid.classify(w[0])
    if language != 'en':
        response = requests.get('https://api.mymemory.translated.net/get?q='+problem+'&langpair='+'ar'+'|en')
        translated = response.json()['responseData']['translatedText']
    else:
        translated=problem
    translated_doc=nlp(translated)
    for sent in translated_doc.sentences:
        sent=sent.text.replace(',','and')
        quants = parser.parse(sent)
        print(quants)
        for q in quants:  
            if q.unit.entity.name != 'dimensionless' :   
                dim_numbers.append(q.value)
    print(dim_numbers)
    for w in res['data']:
        if w[1] == 'NOUN':
            language, _ = langid.classify(w[0])
            if language != 'en':
                response = requests.get('https://api.mymemory.translated.net/get?q='+w[0]+'&langpair='+'ar'+'|en')
                translated = response.json()['responseData']['translatedText']
            else:
                translated=w[0]
            url = "https://api.giphy.com/v1/stickers/search?api_key=iidRVNv0y0mmMUNhYrwlVFufRdIeFLJP&q=" + \
                translated+"&limit=1&offset=1&rating=PG"
            print(url)
            response = requests.get(url)
            if (response.json()['data']):
                w[0] = response.json()['data'][0]['images']['downsized']['url']
            else:w[1]=' '
        if w[1] == 'PROPN': #or w[1] == 'X':
            r2 = requests.get("https://api.genderize.io?name="+w[0])
            gender = r2.json()['gender']
            if gender == 'female':
                w[0] = 'https://media.giphy.com/media/ifMNaJBQEJPDuUxF6n/giphy.gif'
            else:
                w[0] = 'https://media.giphy.com/media/TiC9sYLY9nilNnwMLq/giphy.gif'
        if w[1] == 'ADV':
            language, _ = langid.classify(w[0])
            if language != 'en':
                response = requests.get('https://api.mymemory.translated.net/get?q='+w[0]+'&langpair='+'ar'+'|en')
                translated = response.json()['responseData']['translatedText']
            else:
                translated=w[0]
            url = "https://api.giphy.com/v1/stickers/search?api_key=iidRVNv0y0mmMUNhYrwlVFufRdIeFLJP&q=" + \
                translated+"&limit=1&offset=1&rating=PG"  #W[2] better 
            response = requests.get(url)
            if (response.json()['data']):
                w[0] = response.json()['data'][0]['images']['downsized']['url']

        if (w[1] == 'VERB') and (w[0] in action_verbs):
            url = "https://api.giphy.com/v1/stickers/search?api_key=iidRVNv0y0mmMUNhYrwlVFufRdIeFLJP&q=" + \
                w[0]+"&limit=1&offset=1&rating=PG"
            response = requests.get(url)
            if (response.json()['data']):
                w[0] = response.json()['data'][0]['images']['downsized']['url']
        

    Output_List=[]
    for w in res['data']:
        if (w[1]=='NUM') and (not (int(w[0]) in dim_numbers)) and (int(w[0])<15):
            Output_List.append([w[2],1])
            continue
        if (w[0].startswith('https')):
            Output_List.append([w[0],0])
        else:
            Output_List.append([w[2],0])
   #print(res)
    print("aaaa")
    print(Output_List)
    del doc
    
    return Output_List
import requests
from django.shortcuts import render

def calculate(request):
  if request.method == 'POST':
    problem = request.POST.get('problem')
    role =request.POST.get("role")

    FirstName =request.POST.get("FirstName")
    LastName =request.POST.get("LastName")
    # Appel de votre API pour obtenir le résultat du problème mathématique
    list_output=image_generation(problem)
    for i in range(len (list_output)):
        if list_output[i][1]==1:
            for j in range (int(list_output[i][0])-1):
                list_output.insert(i+1,[list_output[i+1][0],0])
    for i in range(len (list_output)):
      if "http" in list_output[i][0]:
        list_output[i][1]=2
    
            
    result = list_output
    # Renvoi du résultat dans le modèle HTML
    if(role==str(2)): # teacher 
        return render(request, 'index.html', {'result': result,'problem':problem,"role":role})
    else: # student 
        return render(request, 'calculate.html', {'result': result,'problem':problem ,"role":role,
                                                  "FirstName":FirstName,"LastName":LastName
                                                  })

        
  
  else:
    return render(request, 'calculate.html',{"role":role})
def intro(request):
    if request.method == 'POST':
        btn1 = request.POST.get('btn1')
        if(btn1==None):
             return render(request, 'Register.html')
        else:
            return render(request, 'login.html')
    else:
        return render(request, 'intro.html')
##############face recognition#########
import cv2
import time
import pymysql
import numpy as np
from tkinter import *
import education.settings as st
import education.credentials as cr
import face_recognition as f
import education.videoStream as vs
import multiprocessing as mp
from datetime import datetime
from tkinter import messagebox
from playsound import playsound

# The LoginSystem class
class LoginSystem:
    
    def __init__(self, root):
    

        # Declaring a variable with a default value
        self.status = False

     


    # A Function to login into the system through face recognition method 
    def loginEmployee(self):
        # Clear the screen first
        #self.clearScreen()

        # Call a function called playVoice() to play a sound in a different
        # process
        process = self.playVoice("education/Voices/voice1.mp3")
        time.sleep(4)
        # End the process
        process.terminate()
        # Inheriting the class called VideoStream and its
        # methods here from the videoStream module to capture the video stream
        faces = vs.encode_faces()
        encoded_faces = list(faces.values())
        faces_name = list(faces.keys())
        video_frame = True
        # stream = 0 refers to the default camera of a system
        video_stream = vs.VideoStream(stream=0)
        video_stream.start()
        while True:
            if video_stream.stopped is True:
                break
            else :
                
                frame = video_stream.read()

                if video_frame:
                    face_locations = f.face_locations(frame)
                    unknown_face_encodings = f.face_encodings(frame, face_locations)
                    
                    face_names = []
                    for face_encoding in unknown_face_encodings:
                        # Comapring the faces
                        matches = f.compare_faces(encoded_faces, \
                        face_encoding)
                        name = "Unknown"

                        face_distances = f.face_distance(encoded_faces,\
                        face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = faces_name[best_match_index]
                        face_names.append(name)
                    

                video_frame = not video_frame

                for (top, right, bottom, left), faceID in zip(face_locations,\
                face_names):
                    
                    # Draw a rectangular box around the face
                    cv2.rectangle(frame, (left-20, top-20), (right+20, \
                    bottom+20), (0, 255, 0), 2)
                    # Draw a Label for showing the name of the person
                    cv2.rectangle(frame, (left-20, bottom -15), \
                    (right+20, bottom+20), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    # Showing the face_id of the detected person through 
                    # the WebCam
                    cv2.putText(frame, "Face Detected", (left -20, bottom + 15), \
                    font, 0.85, (255, 255, 255), 2)
                    

                    # Call the function for attendance
                    self.status = self.isPresent(faceID)
            # delay for processing a frame 
            delay = 0.04
            time.sleep(delay)

            cv2.imshow('frame' , frame)
            key = cv2.waitKey(1)
            # If self.status is True(which means the face is identified)
            # a voice will play in the background, the look will be break,
            # and all cv2 window will be closed.
            if self.status == True:
                print('mawjoud')
                process = self.playVoice("education/Voices/voice2.mp3")
                time.sleep(2)
                process.terminate()
                break
        video_stream.stop()

        # closing all windows 
        cv2.destroyAllWindows()
        # Calling a function to show the status after entering an employee
        #self.employeeEntered()

    # A Function to check if the user id of the detected face is matching 
    # with the database or not. If yes, the function returns the value True.
    def isPresent(self, UID):
        
        try:
           
            students = Student.objects.all()
            student_serializer = StudentSerializer(students, many=True)
            list_students=student_serializer.data
            teachers = Teacher.objects.all()
            teacher_serializer = TeacherSerializer(teachers, many=True)
            list_teacher=teacher_serializer.data      
            listall=list_students+list_teacher
            
            dict_list_all = [dict(item) for item in listall]
            for dictionary in dict_list_all:
                if str(dictionary.get('id')) == str(UID):
                    
                    return True
                else:
                    pass
            

        except Exception as e:
                print(str(e))
                #messagebox.showerror("Error!",f"Error due to {str(e)}",parent=self.window)

    # A Function to play voice in a different process
    def playVoice(self, voice):
        process = mp.Process(target=playsound, args=(voice,))
        process.start()
        return process

from io import BytesIO
def save_image_from_base64(image_data, directory, filename):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, filename)
        with open(file_path, "wb") as f:
            f.write(image_data)
        return file_path
    except Exception as e:
        print(str(e))
        return None
def save_images(list,name):
    for dictionary in list:
        b64_string = str(dictionary.get('picture'))
        # decode the base64 string to binary image data
        binary_data = base64.b64decode(b64_string)
        # open the image using PIL
        save_image_from_base64(binary_data, "education/Images"+name, str(dictionary.get('id'))+".png")
def login_employee(request):
    role=0
    students = Student.objects.all()
    student_serializer = StudentSerializer(students, many=True)
    list_students=student_serializer.data
    teachers = Teacher.objects.all()
    teacher_serializer = TeacherSerializer(teachers, many=True)
    list_teacher=teacher_serializer.data  
    listall=list_students +list_teacher
    
        
    if request.method == 'POST':
        Button = request.POST.get('button') 
        login_system = LoginSystem(root=None)
        if(Button==None):
            return render(request, 'login.html')
        else:          
            save_images(listall,"")
            # Call the loginEmployee function
            login_system.loginEmployee()
            if login_system.status:
                #os.rmdir("education/Images")
                #If the face is identified, redirect to the success page
                return render(request, 'calculate.html',{"role":1})
            else:
                return render(request, 'login.html')
    else:
        return render(request, 'login.html')
def Tik_Tac_Toe(request):
    if request.method == 'POST':
         return render(request, 'Tik_Tac_Toe.html')
    else:
         return render(request, 'calculate.html')
def MathQuiz(request):
    if request.method == 'POST':
         return render(request, 'ttt.html')
    else:
         return render(request, 'calculate.html')
def RockPaperScissors(request):
    if request.method == 'POST':
         return render(request, 'game.html')
    else:
         return render(request, 'calculate.html')
def LogOut(request):
    if request.method == 'POST':
         return render(request, 'login.html')
    else:
         return render(request, 'calculate.html')
def MemoryGame(request):
    if request.method == 'POST':
         return render(request, 'memory.html')
    else:
         return render(request, 'calculate.html')    

def login_user(request):
    if request.method=='POST':
        role=0
        email = request.POST.get('email')
        password=request.POST.get('password')
        students = Student.objects.all()
        student_serializer = StudentSerializer(students, many=True)
        list_students=student_serializer.data
        dict_list_student= [dict(item) for item in list_students]
        for dictionary in dict_list_student:
            if (str(dictionary.get('email')) == str(email) and (str(dictionary.get('password')) == str(password))):
                role=1 #role=1 si le user est un student
                return render(request, 'calculate.html',
                              {"role":role,"FirstName":str(dictionary.get('FirstName')),
                               "LastName":str(dictionary.get('LastName'))})
        teachers = Teacher.objects.all()
        teacher_serializer = TeacherSerializer(teachers, many=True)
        list_teacher=teacher_serializer.data      
        dict_list_teacher = [dict(item) for item in list_teacher]
        for dictionary in dict_list_teacher:
            if (str(dictionary.get('email')) == str(email) and (str(dictionary.get('password')) == str(password))):
                role=2#role=1 si le user est un teacher  
                return render(request, 'calculate.html',
                              {"role":role,"FirstName":str(dictionary.get('FirstName')),
                               "LastName":str(dictionary.get('LastName'))})
            
        messages.error(request, 'login password or email incorrect')

        
        return redirect('login_user')        
    else:
        return render(request, 'login.html')
def email_existe(email):
    students = Student.objects.all()
    student_serializer = StudentSerializer(students, many=True)
    list_students=student_serializer.data
    teachers = Teacher.objects.all()
    teacher_serializer = TeacherSerializer(teachers, many=True)
    list_teacher=teacher_serializer.data      
    listall=list_students+list_teacher
    dict_list_all = [dict(item) for item in listall]
    for dictionary in dict_list_all:
        if str(dictionary.get('email')) == str(email) :
            return True
    return False
def class_existe(Class):
    Classrooms = Class.objects.all()
    Classroom_serializer = ClassSerializer(Classrooms, many=True)
    list_Classrooms=Classroom_serializer.data
    dict_list_all = [dict(item) for item in list_Classrooms]
    for dictionary in dict_list_all:
        if str(dictionary.get('Class')) == str(Class) :
            return True
    return False

import base64


def register_user(request):
    if request.method=='POST':
        email = request.POST.get('email')
        birthdate =request.POST.get('birthdate')
        password = request.POST.get('password')
        FirstName = request.POST.get('FirstName')
        LastName = request.POST.get('LastName')
        picture = request.FILES['picture']
        file_content = picture.read()
        b64_content = base64.b64encode(file_content).decode('utf-8')
        
        if (email_existe(email)):
            messages.error(request, 'email exist')
            return redirect('register_user') 
        #elif not (class_existe(Class)):
        #    messages.error(request, "class doesn't exist")
         #   return redirect('register_user') 

        else :
            user_data =    {
            "password": password,
            "email": email,
            "FirstName": FirstName,
          "LastName": LastName,
           "picture": b64_content
         }
        birthdate = datetime.strptime(birthdate, "%Y-%m-%d").date()

        dob=birthdate
        
        today = date.today()
        
        age = today.year - dob.year -((today.month, today.day) <(dob.month, dob.day))
        if (5<age<12):
            student_serializer = StudentSerializer(data=user_data)
            if student_serializer.is_valid():
                student_serializer.save()
                messages.success(request, 'Registration successful!')
                return redirect('login_user')
            else:
                messages.error(request, 'Registration failed. Please try again.')
                return redirect('register_user')
        elif(50>age>12) :
            teacher_serializer = TeacherSerializer(data=user_data)
            if teacher_serializer.is_valid():
                teacher_serializer.save()
                messages.success(request, 'Registration successful!')
                return redirect('login_user')
            else:
                messages.error(request, 'Registration failed. Please try again.')
                return redirect('register_user')
    else:
        return render(request, 'register.html')
from django.shortcuts import render
from django.conf import settings
import json
@csrf_exempt

def chatbot_api(request):
    chatbot_response =None 
    api_key = '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'
    if api_key is not None and  request.method=="POST" :
        openai.api_key=api_key
        user_input =request.POST.get('user_input')
        prompt =user_input
        response =openai.Completion.create(
            engine='text-davinci-003',
            prompt='answer this like a chatbot for a web application:'+ prompt,
            max_tokens=256,
            #stop="."
            temperature=0.5
        )

        
        chatbot_response = response["choices"][0]["text"]
        return JsonResponse({'response': chatbot_response})
    else:
        return render(request, 'login.html')
#################################Classroom#################################################""
def code_class_existe(code):
    Classrooms = Class.objects.all()
    Classroom_serializer = ClassSerializer(Classrooms, many=True)
    list_Classrooms=Classroom_serializer.data
    dict_list_all = [dict(item) for item in list_Classrooms]
    print(dict_list_all)
    for dictionary in dict_list_all:
        print(str(dictionary.get('code')))
        print(code)
        if str(dictionary.get('code')) == str(code) :
            return True
    return False
import base64
import pyautogui
import io
from PIL import Image
import time
def save_screenshot(request):
    print("a")
    if request.method == 'POST':
        print("a")
        Code = request.POST.get('Code')
        ###
        ss = pyautogui.screenshot()
        width, height = ss.size
        left = 450
        top = 270
        right = width - 600
        bottom = height - 550
        cropped = ss.crop((left, top, right, bottom))
        img_bytes = io.BytesIO()
        cropped.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
        classes = Class.objects.all()
        class_serializer = ClassSerializer(classes, many=True)
        list_classes=class_serializer.data
        dict_list_class= [dict(item) for item in list_classes]
        for dictionary in dict_list_class:
            if (str(dictionary.get('code')) == str(Code)):
                class_found = dictionary
        # Save the screenshot data to a file or a database
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print("naraaa el code:")
        print(Code)
        if code_class_existe(Code):
            problem =    {
            "base64image": img_b64,
           "Class": class_found.get('id')
         }
            problem_serializer = ProblemSerializer(data=problem)
            print(problem_serializer)
            if problem_serializer.is_valid():
                print("aaaaaaaaaa")
                print(pyautogui.size())
                problem_serializer.save()
            print(problem_serializer.is_valid())
            print(class_found.get('id'))
            return render(request, 'calculate.html')
    else:
        return render(request, 'index.html')
    
###################################GET CODE AND DISPLAY PROBLEMS IN CLASSROOM#####################################
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def getClassProblems(request):
    if request.method=='POST':
        code = request.POST.get('Code')
        problems = Problem.objects.filter(Class__code=code) # filter problems by class code
        problem_serializer = ProblemSerializer(problems, many=True)
        list_problems = problem_serializer.data
        base64images = [image["base64image"] for image in list_problems]
        print(list_problems)
        return render(request, 'classroom.html', {'problems': base64images})
    else:
        return render(request, 'get_code_classroom.html')



###################################GET CODE AND ADD CLASSROOM#####################################
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def addClassroom(request):
    if request.method=='POST':
        code = request.POST.get('Code')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # get teacher object by email and password
        try:
            teacher = Teacher.objects.get(email=email, password=password)
        except Teacher.DoesNotExist:
            return HttpResponse('Teacher not found', status=404)

        # create new class object with given code and teacher
        new_class = Class.objects.create(code=code, teacher=teacher)

        # return success message
        return render(request, 'get_code_classroom.html')

        # handle invalid request method
    return render(request, 'create_classroom.html')    

            


######################DEEP FAKE VIDEOOOOOOO##################################################
from django.shortcuts import render, redirect, reverse
@csrf_exempt
def deepFakeVideo(request):
    if request.method=='POST':
        url = "https://api.d-id.com/talks"
        api_key='YWJvdWJha2VyLnJlYmFpQGVzcHJpdC50bg:ePCCr5rnFaDCPcHXpxulW'
        auth_string = base64.b64encode(f'{api_key}:'.encode()).decode()
        a=request.POST.get('scripts')
        payload = {
            "script": {
                "type": "text",
                "provider": {
                    "type": "microsoft",
                    "voice_id": "en-US-BrandonNeural",
                    "voice_config": {
                        "style": "string",
                        "rate": "1",
                        "pitch": "+2st"
                    }
                },
                "input": a
            },
            "config": {
                "stitch": True,
            },

            "source_url": "https://create-images-results.d-id.com/auth0%7C645956f4151ddf91659eac96/upl_VecVAgjHCruHxw-iPh9cr/image.png"
            #"webhook": "https://host.domain.tld/to/webhook"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            'Authorization': f'Basic {auth_string}',
        }

        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        id = response_data['id']
        print(id)
        #request.session['id'] = id
        '''url = f"https://api.d-id.com/talks/{id}"
        headers = {"accept": "application/json",
                   "content-type": "application/json",
                   'Authorization': f'Basic {auth_string}'}
        response1 = requests.get(url, headers=headers)
        response_data1 = response1.json()
        print("aaaaaaaaaaaaaaaaaaaaaaa")
        print(response1.text)'''
        return  render(request,'calculate.html',{'id': id})

    
@csrf_exempt
def deepFakeVideoGet(request, id):
    if request.method=='GET':
        #id = request.session.get('id')
        api_key='YWJvdWJha2VyLnJlYmFpQGVzcHJpdC50bg:ePCCr5rnFaDCPcHXpxulW'
        auth_string = base64.b64encode(f'{api_key}:'.encode()).decode()
        url = f"https://api.d-id.com/talks/{id}"
        headers = {"accept": "application/json",
                "content-type": "application/json",
                'Authorization': f'Basic {auth_string}'}
        response1 = requests.get(url, headers=headers)
        response_data1 = response1.json()
        result_url = response_data1.get('result_url') 
        print("aaaaaaaaaaaaaaaaaaaaaaa")
        print(response_data1)
        print(id)
        return render(request, 'video.html',{'result_url':result_url})
    



@csrf_exempt
def respond(request):
    if request.method=='POST':
        problem = request.POST.get('input_problem')
        return render(request,'answer.html',{'problem':problem})
    else:
        return render(request,'answer.html',{'problem':problem})    


@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def solution(request):
    if request.method=='POST':
        problem = request.data.get('problem')
        student_response = request.data.get('answer')
        openai.api_key=api_key
        prompt ="This is a simple math problem: "+ problem + " This is my response to the problem : "+ student_response + "Is my response correct? " 
        response =openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=256,
            #stop="."
            temperature=0.5
        )
        print(response)
        chatbot_response=response["choices"][0]["text"]
        return JsonResponse({'chatbot_response': chatbot_response})
            
      