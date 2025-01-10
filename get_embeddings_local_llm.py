import logging
import requests
import subprocess  # Add this line to import subprocess
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LM_STUDIO_URL = "https://ngrok.wrench.chat/v1/embeddings"

# Initialize BigQuery client
client = bigquery.Client()

def get_token():
    """Retrieve the authentication token using gcloud."""
    command = "gcloud auth print-identity-token"
    return subprocess.check_output(command, shell=True).decode("utf-8").strip()

def get_embedding(text):
    """Fetch embeddings from LM Studio for a given text."""
    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json"
    }
    payload = {"input": [text]}
    
    try:
        response = requests.post(LM_STUDIO_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [{}])[0].get('embedding', [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get embedding: {e}")
        return None

def fetch_unembedded_rows(client):
    """Fetch rows from the AvaKeyWords table where embeddings are missing (NULL or empty)."""
    query = """
    SELECT Name, Content
    FROM `looker-tickets.zendesk.AvaKeyWords`
    WHERE Embedding IS NULL OR ARRAY_LENGTH(Embedding) = 0
    """
    return client.query(query).result()

def update_embeddings(client, rows_to_update):
    """Update BigQuery with the generated embeddings."""
    table_id = 'looker-tickets.zendesk.AvaKeyWords'
    rows_to_update = [
        {
            'Name': row['Name'],
            'Embedding': row['Embedding']
        } for row in rows_to_update
    ]

    errors = client.insert_rows_json(table_id, rows_to_update)
    if errors:
        logging.error(f"Failed to insert embeddings: {errors}")
    else:
        logging.info(f"Successfully inserted {len(rows_to_update)} rows into BigQuery.")

def process_rows(client):
    """Process all rows, generate embeddings, and update BigQuery."""
    rows = fetch_unembedded_rows(client)
    rows_to_update = []
    
    for row in rows:
        embedding = get_embedding(row['Content'])
        if embedding:
            rows_to_update.append({
                'Name': row['Name'],
                'Embedding': embedding
            })
    
    if rows_to_update:
        update_embeddings(client, rows_to_update)
    else:
        logging.warning("No valid embeddings generated for the records.")

if __name__ == "__main__":
    try:
        logging.info("Starting embedding generation process.")
        process_rows(client)
    except Exception as e:
        logging.error(f"An error occurred: {e}")