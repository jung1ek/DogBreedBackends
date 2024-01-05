from django.http import HttpResponse
from django.http import JsonResponse
from openai import OpenAI
import os
import openai
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

client = openai.OpenAI(api_key='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')

assistant = client.beta.assistants.create(
    name = "Dog Care Assistant",
    instructions = "You are a knowledgeable dog care assistant. you are here to provide helpful "+
        "information about various dog breeds, "+
            "training tips, health advice, and "+
                "general canine care according to user questions. "
                     "Also answer the every dog related questions. "
    "Retrive every solution from the reliable "
    "website like www.britannica.com, www.akc.org and soon. also, "
        "avoid every toxic, harmful and unrelated to dog questions.",
    description = "Dog Care Assistant - Providing information on dog breeds, training tips, and health advice",
    # tools = [{'type':'function'}],
    model = 'gpt-3.5-turbo-1106'
)

thread = client.beta.threads.create()



#json_data = messages['choices'][0]['text']
@csrf_exempt
@api_view(['GET'])
def retrive(request):
    message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="i have golden retriever. what are the characteristics of my dog"
    )
    
    run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
                instructions="Please address the user as Jane Doe"
    ) 
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id)
        if run.status=='completed':
            break
    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )
    message_data = messages.data[0].content[0].text.value
    return JsonResponse({'system':message_data})
    
