import openai
import os

def check_openai_api_key():
    client = openai.OpenAI(api_key="")
    try:
        print()
    except openai.AuthenticationError:
        return False
    else:
        return True

def get_openai_response():
    client = openai.OpenAI(api_key="api-key")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What is 2 + 3?"}
        ]
    )

    return response.choices[0].message.content.strip()
# print(os.getenv("OPENAI_API_KEY"))
print(get_openai_response())