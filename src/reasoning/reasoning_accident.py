
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    reasoning={
        "effort": "medium"
    },
    input="Explain what a red traffic light means for a vehicle at an intersection in NYC."
)

print(response.output_text)


