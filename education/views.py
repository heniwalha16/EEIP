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
################################ ClassApi   #####################################

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

action_verbs = ['Provide', 'Zip', 'Train', 'Clean', 'Drink', 'Face', 'Build', 'Control', 'Release', 'Have', 'Play', 'Press', 'Turn', 'Sit down', 'Smell', 'Arrange', 'Ski', 'Result', 'Wonder', 'Expect', 'Visit', 'State', 'Explain', 'Correct', 'Increase', 'Wait', 'Repeat', 'Bathe', 'Run', 'Tell', 'Know', 'See', 'Mind', 'Own', 'Throw away', 'Complain', 'Feel', 'Affect', 'Buy', 'Do', 'Hug', 'Record', 'Replace', 'Sit', 'Plan', 'Admit', 'Invite', 'Pay', 'Try', 'Relate', 'Invent', 'Tend', 'Turn on', 'Order', 'Coach', 'Deliver', 'Limit', 'Apply', 'Reduce', 'Yank', 'Accept', 'Survive', 'Influence', 'Color', 'Remember', 'Form', 'Wash', 'Start', 'Describe', 'Measure', 'Share', 'Climb', 'Cough', 'Involve', 'Touch', 'Suppose', 'Keep', 'Cook', 'Approve', 'Inform', 'Produce', 'Skip', 'Shout', 'Agree', 'Suggest', 'Achieve', 'Offer', 'Cost', 'Arrive', 'Kiss', 'Afford', 'Last', 'Could', 'Understand', 'Protect', 'Answer', 'Stand', 'Point', 'Go', 'Check', 'Happen', 'Exist', 'Receive', 'Rise', 'Collect', 'Stand up', 'Ask', 'Enter', 'Continue', 'Damage', 'Fall', 'Contain', 'Remove', 'Scream', 'Believe', 'Clap', 'Come', 'Fly', 'Whistle', 'Destroy', 'Sing', 'Teach', 'Perform', 'Listen', 'Sneeze', 'Win', 'Supply', 'Leave', 'Enjoy', 'Edit', 'Reach', 'Experience', 'Must', 'Dream', 'Avoid', 'Paint', 'Shake', 'Set', 'Develop', 'Deal', 'Learn', 'Stack', 'Get', 'Carry', 'Follow', 'Speak', 'Dive', 'Write', 'Eat', 'Jump', 'Hold', 'Shop', 'Drive', 'Turn off', 'Show', 'Forgive', 'Live', 'Treat', 'Snore', 'Use', 'Make', 'Express', 'Finish', 'Forget', 'Cut', 'Move', 'Watch', 'Draw', 'Lie', 'Watch TV', 'Regard', 'Discover', 'Improve', 'Deny', 'Allow', 'Smile', 'Bow', 'Love', 'Dance', 'Hope', 'Prevent', 'Argue', 'Fight', 'Need', 'Shoot', 'Succeed', 'Meet', 'Consist', 'Choose', 'Grow', 'Take', 'Lend', 'Walk', 'Open', 'Give', 'Reply', 'Exit', 'Dig', 'Travel', 'Change', 'Think', 'Ride', 'Reveal', 'Identity', 'Return', 'Depend', 'Like', 'Matter', 'Close', 'Become', 'Create', 'Break', 'Send', 'Laugh', 'Cry', 'Hear', 'Encourage', 'Cause', 'Sound', 'Dress', 'Look', 'Say', 'Prefer', 'Care', 'Report', 'Help', 'Call',"Find", "Cross", "Save", "Imitate", "Sleep", "Clear", "Contribute", "Prepare", "Imagine", "Begin", "Crawl", "Solve", "Push", "Sew", "Study", "Mention", "Mean", "Join", "Complete", "Throw", "Read", "Act", "Disappear", "Catch", "Hide", "Knit", "Sell", "Talk", "Want"]
action_verbs = [word.lower() for word in action_verbs]

@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def image_generation(request):
    if request.method == 'POST':
        print("aaa")
        seed = request.POST.get('problem')
    # This sets up a default neural pipeline in Lang
    print(seed)
    lang=detect_language(seed)
    if (lang != 'en'):
        response = requests.get('https://api.mymemory.translated.net/get?q='+seed+'&langpair='+lang+'|en')
        seed = response.json()['responseData']['translatedText']
    if ',' in seed:
        seed=seed.replace(',',' , ')
    nlp = stanza.Pipeline('en', use_gpu=False,
                          processors='tokenize,pos,lemma')
    doc = nlp(seed)
    res = {'type': None, 'data': []}

    # Load the trained model and tokenizer
    model = BertForMathProblemClassification()
    model.load_state_dict(torch.load('C:/Users/user/Downloads/bert_math_problem_classification.pt'))
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # Example math problem
    problem = seed

    # Tokenize the input and convert to tensors
    input_ids = torch.tensor(tokenizer.encode(problem, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    attention_mask = torch.ones_like(input_ids)

    # Pass the input to the model and get the predicted class
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs).item()

    # Print the predicted class
    class_names = ['Not Geometry', 'Geometry']
    problem_type = class_names[predicted_class]
    
    if (problem_type=='Geometry'):      
        
        for sent in doc.sentences:
            quants = parser.parse(sent.text)
            for q in quants:
                if q.unit.entity.name == 'length' :
                    res['data'].append([float(q.surface.split()[0]), q.surface.split()[1]])
                    
        if len(res['data']) > 0:
            if len(res['data'])==1:
                if 'diameter' in problem:
                    res['type'] = 'diametre'
                elif 'radius' in problem:
                    res['type'] = 'radius'
                else:
                    res['type'] = 'square'
            elif len(res['data'])==2:
                if 'parallelogram' in problem:
                    res['type'] = 'parallelogram'
                elif 'rhombus' in problem:
                    res['type'] = 'rhombus'
                else:
                    res['type'] = 'rectangle'
            elif len(res['data'])==4:
                if 'trapezium' in problem:
                    res['type'] = 'trapezium'
            del doc
            return res


    else:
        res['type'] = 'entity'

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
        if w[1] == 'NOUN':
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
    print(res)
    dim_numbers=[]
    for sent in doc.sentences:
        sent=sent.text.replace(',','and')
        quants = parser.parse(sent)
        
        for q in quants:  
            if q.unit.entity.name != 'dimensionless' :   
                dim_numbers.append(q.value)
    print(dim_numbers)
    for w in res['data']:
        if w[1] == 'NOUN':
            url = "https://api.giphy.com/v1/stickers/search?api_key=iidRVNv0y0mmMUNhYrwlVFufRdIeFLJP&q=" + \
                w[0]+"&limit=1&offset=1&rating=PG"
            response = requests.get(url)
            if (response.json()['data']):
                w[0] = response.json()['data'][0]['images']['downsized']['url']
            else:w[1]='NOUN_'
        if w[1] == 'PROPN': #or w[1] == 'X':
            r2 = requests.get("https://api.genderize.io?name="+w[0])
            gender = r2.json()['gender']
            if gender == 'female':
                w[0] = 'https://media.giphy.com/media/ifMNaJBQEJPDuUxF6n/giphy.gif'
            else:
                w[0] = 'https://media.giphy.com/media/TiC9sYLY9nilNnwMLq/giphy.gif'
        if w[1] == 'ADV':
            url = "https://api.giphy.com/v1/stickers/search?api_key=iidRVNv0y0mmMUNhYrwlVFufRdIeFLJP&q=" + \
                w[0]+"&limit=1&offset=1&rating=PG"  #W[2] better 
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
        if (w[1]=='NUM') and (int(w[0]) in dim_numbers) and (int(w[0])<15):
            Output_List.append([w[2],1])
            continue
        if (w[0].startswith('https')):
            Output_List.append([w[0],0])
        else:
            Output_List.append([w[2],0])
   #print(res)
    #print("aaaa")
    #print(Output_List)
    del doc
    return HttpResponse(Output_List)
