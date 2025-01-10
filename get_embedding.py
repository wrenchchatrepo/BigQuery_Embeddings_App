import openai

# Set up the OpenAI client to point to LM Studio locally using your custom ngrok domain
client = openai.OpenAI(base_url="https://ngrok.wrench.chat", api_key="lm-studio")

def get_embedding(text, model="second-state/All-MiniLM-L6-v2-Embedding-GGUF"):
    text = text.replace("\n", " ")

    # Assuming the embeddings endpoint and structure match OpenAI's API style
    response = client.embeddings.create(input=[text], model=model)
    
    # Return the first embedding from the response
    return response["data"][0]["embedding"]

# Example usage
embedding = get_embedding("Once upon a time, there was a cat.")
print(embedding)