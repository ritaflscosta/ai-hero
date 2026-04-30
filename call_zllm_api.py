import os
from openai import OpenAI

# Uncomment these lines if you're in datalab
# from zdatalab import get_ztoken

_client = None

def initialize_zllm(ztoken=None):
    """Initialize the zLLM client.
    
    Args:
        ztoken: zToken for authentication. If not provided, uses ZTOKEN environment variable.
    
    Returns:
        OpenAI client instance configured for zLLM.
    """
    global _client
    
    if ztoken is None:
        ztoken = os.environ.get("ZTOKEN")
    
    if not ztoken:
        raise ValueError("ztoken must be provided or set in ZTOKEN environment variable")
    
    # by changing the base url you are calling zLLM and not OpenAI
    _client = OpenAI(
        api_key=ztoken,
        base_url="https://zllm.data.zalan.do/v1/"
    )
    
    return _client

def get_client():
    """Get the zLLM client, initializing if necessary."""
    global _client
    
    if _client is None:
        initialize_zllm()
    
    return _client

def llm(prompt, model="bedrock/anthropic.claude-sonnet-4-6"):
    """Call the zLLM API with a prompt.
    
    Args:
        prompt: The prompt to send to the model.
        model: The model to use (default: bedrock/anthropic.claude-sonnet-4-6).
    
    Returns:
        The model's response as a string.
    """
    client = get_client()
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message.content