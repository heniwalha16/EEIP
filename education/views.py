from django.shortcuts import render
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
