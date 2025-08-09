# Use even smaller model - create this file:
# ultra_light_demo.py

from llama_cpp import Llama

# Download manually first
import requests
url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
response = requests.get(url)
with open("models/tinyllama_q2.gguf", "wb") as f:
    f.write(response.content)

# Load with minimal memory
llm = Llama(
    model_path="models/tinyllama_q2.gguf",
    n_ctx=512,  # Small context
    n_batch=8,  # Small batch
    n_threads=4  # Few threads
)

# Test
response = llm("What is 2+2?", max_tokens=20)
print(response['choices'][0]['text'])