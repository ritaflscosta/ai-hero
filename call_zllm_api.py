import argparse
import os
from openai import OpenAI

# Uncomment these lines if you're in datalab
# from zdatalab import get_ztoken

# Get ztoken from command-line argument or environment variable
parser = argparse.ArgumentParser(description="Call zLLM API")
parser.add_argument("--token", type=str, help="zToken for authentication")
args = parser.parse_args()

ztoken = args.token or os.environ.get("ZTOKEN")
if not ztoken:
    raise ValueError("ztoken must be provided via --token argument or ZTOKEN environment variable")

# by changing the base url you are calling zLLM and not OpenAI
client = OpenAI(
    api_key=ztoken,
    base_url="https://zllm.data.zalan.do/v1/"
)
# client.models.list() # list models

def llm(prompt, model="bedrock/anthropic.claude-sonnet-4-6"):
    messages=[
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = llm(prompt)
    print(response)