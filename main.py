import requests
from flask import Flask, request, jsonify

def get_embedding(request):
    data = request.get_json()
    text = data.get('text', '')

    # Use the ngrok URL instead of localhost
    lm_studio_url = "https://ngrok.wrench.chat/v1/embeddings"

    response = requests.post(lm_studio_url, json={
        "input": [text],
        "model": "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
    })

    if response.status_code == 200:
        embedding = response.json().get('data', [{}])[0].get('embedding', [])
        return jsonify({"embedding": embedding})
    else:
        return jsonify({"error": "Failed to get embeddings"}), response.status_code