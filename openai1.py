from openai import OpenAI
import os
import openai

CLIENT = OpenAI(api_key='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')

straem=CLIENT.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user","content":"What does the fox say?"}
        ]
)
print(straem.choices[0].message)