import requests
import logging
from flask import Flask, request, jsonify

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def root_handler():
    logging.debug("Received request at root endpoint")
    return get_embedding()

# Endpoint to handle embedding requests
@app.route('/get-embedding', methods=['POST'])
def get_embedding():
    logging.debug("Received request at /get-embedding endpoint")
    data = request.json
    logging.debug(f"Received data: {data}")
    text = data.get('text', '')
    
    # LM Studio API URL for embeddings (updated to use your custom domain)
    lm_studio_url = "https://ngrok.wrench.chat/v1/embeddings"
    
    try:
        # Call LM Studio API to get embeddings
        logging.debug(f"Sending request to LM Studio at {lm_studio_url}")
        response = requests.post(lm_studio_url, json={
            "input": [text],
            "model": "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
        })
        
        logging.debug(f"LM Studio response status: {response.status_code}")
        logging.debug(f"LM Studio response content: {response.text}")
        
        if response.status_code == 200:
            embedding = response.json().get('data', [{}])[0].get('embedding', [])
            return jsonify({"embedding": embedding})
        else:
            error_message = f"Failed to get embeddings. LM Studio response: {response.text}"
            logging.error(error_message)
            return jsonify({"error": error_message}), response.status_code
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    logging.info("Starting Flask app")
    app.run(host='0.0.0.0', port=8080)