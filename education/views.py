from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt   
from rest_framework import serializers, viewsets 
from rest_framework.decorators import action 
from rest_framework.response import Response  
from django.views.decorators.csrf import csrf_exempt 
from rest_framework.decorators import api_view, renderer_classes 
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
import openai ,os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("OPENAI_KEY",None)
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