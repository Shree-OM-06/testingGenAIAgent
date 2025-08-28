import os
import json
import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

def gpt_40(user_prompt: str):
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("DEPLOYMENT_NAME")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("API_VERSION")

    # Initialize Azure OpenAI client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
        http_client=httpx.Client(verify=False)  # disable SSL verification if needed
    )

    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an AI assistant that helps people find information."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    # Generate the response
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=160,
        temperature=0.7
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stream=False
    )

    result = json.loads(completion.to_json())
    answer = result["choices"][0]["message"]["content"]
    print("Response:", answer)
    return answer


if __name__ == "__main__":
    user_input = "Explain the theory of relativity in simple terms."
    gpt_40(user_input)
