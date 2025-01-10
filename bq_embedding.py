"""
BigQuery Embedding Script

This script processes conversation texts from a BigQuery table, generates embeddings for them,
and updates the table with the new embeddings. It uses batch processing for improved performance,
processing records in descending order of ticket_id.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from collections import Counter
import argparse
import subprocess
from importlib import metadata, import_module

# Ensure Python version is 3.8 or later
if sys.version_info < (3, 8):
    print("This script requires Python 3.8 or later.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required packages
required_packages = ['requests', 'google.cloud.bigquery', 'tqdm']
missing_packages = []

for package in required_packages:
    try:
        print(f"Attempting to import {package}...")
        if '.' in package:
            module = import_module(package)
        else:
            module = __import__(package)
        print(f"Successfully imported {package}")
        print(f"{package} path: {module.__file__}")
    except ImportError as e:
        print(f"Failed to import {package}. Error: {e}")
        missing_packages.append(package)

if missing_packages:
    logger.error(f"The following required packages are missing: {', '.join(missing_packages)}")
    logger.info("Please install the missing packages using the following command:")
    logger.info(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Third-party imports
import requests
from google.cloud import bigquery
from google.api_core import exceptions
from tqdm import tqdm

CHECKPOINT_FILE = 'ticket_id_checkpoint.txt'
BATCH_SIZE = 1  # Number of rows to fetch from BigQuery in each query
UPDATE_BATCH_SIZE = 1  # Number of rows to accumulate before updating BigQuery
TOKEN_CACHE_FILE = 'token_cache.txt'
TOKEN_EXPIRY_TIME = 3600  # Token expiry time in seconds (1 hour)
MAX_CONCURRENT_REQUESTS = 10  # Maximum number of concurrent API requests

# Counter for operations
operation_counter = Counter()

def get_token():
    if os.path.exists(TOKEN_CACHE_FILE):
        with open(TOKEN_CACHE_FILE, 'r') as f:
            cached_token, expiry_time = f.read().split(',')
        if datetime.now() < datetime.fromisoformat(expiry_time):
            return cached_token

    command = "gcloud auth print-identity-token"
    token = subprocess.check_output(command, shell=True).decode("utf-8").strip()
    expiry_time = (datetime.now() + timedelta(seconds=TOKEN_EXPIRY_TIME)).isoformat()
    
    with open(TOKEN_CACHE_FILE, 'w') as f:
        f.write(f"{token},{expiry_time}")
    
    return token

def get_embedding(text):
    url = "https://ngrok.wrench.chat/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json"
    }
    payload = {"input": [text]}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        operation_counter['post'] += 1
        return data.get('data')[0].get('embedding')
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get embedding: {e}")
        return None

def create_temp_table(client, temp_table_id):
    schema = [
        bigquery.SchemaField("ticket_id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("embeddings", "FLOAT64", mode="REPEATED")
    ]
    table = bigquery.Table(temp_table_id, schema=schema)
    
    try:
        client.delete_table(temp_table_id)
        logging.info(f"Deleted existing temporary table {temp_table_id}")
    except exceptions.NotFound:
        pass
    
    table = client.create_table(table)
    logging.info(f"Created temporary table {temp_table_id}")
    return table

def update_embeddings(client, embeddings_data, dry_run=False):
    table_id = 'looker-tickets.zendesk.conversations_update_deduped_clean'
    temp_table_id = 'looker-tickets.zendesk.temp_embeddings_update'

    if dry_run:
        logging.info(f"Dry run: Would update {len(embeddings_data)} rows in BigQuery")
        return len(embeddings_data)

    # Create a temporary table for the new data
    temp_table = create_temp_table(client, temp_table_id)

    rows_to_update = [
        {"ticket_id": ticket_id, "embeddings": embedding}
        for ticket_id, embedding in embeddings_data
    ]

    # Insert data into the temporary table with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            errors = client.insert_rows_json(temp_table, rows_to_update)
            if errors:
                raise Exception(f"Errors occurred while inserting rows into temp table: {errors}")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to insert rows after {max_retries} attempts: {e}")
                return 0
            logging.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    # Perform the MERGE operation
    merge_query = f"""
    MERGE `{table_id}` T
    USING `{temp_table_id}` S
    ON T.ticket_id = S.ticket_id
    WHEN MATCHED THEN
      UPDATE SET T.embeddings = S.embeddings
    WHEN NOT MATCHED THEN
      INSERT (ticket_id, embeddings)
      VALUES (S.ticket_id, S.embeddings)
    """
    query_job = client.query(merge_query)
    query_job.result()

    # Verify the update
    ticket_ids = [row['ticket_id'] for row in rows_to_update]
    verify_query = f"""
    SELECT COUNT(*) as updated_count
    FROM `{table_id}`
    WHERE ticket_id IN UNNEST({ticket_ids})
    AND ARRAY_LENGTH(embeddings) > 0
    """
    query_job = client.query(verify_query)
    results = query_job.result()
    updated_count = next(iter(results)).updated_count

    if updated_count == len(rows_to_update):
        operation_counter['update'] += updated_count
        logging.info(f"Successfully verified update of {updated_count} rows in BigQuery")
    else:
        logging.warning(f"Mismatch in update count. Attempted: {len(rows_to_update)}, Verified: {updated_count}")

    # Clean up the temporary table
    client.delete_table(temp_table_id)

    return updated_count

def get_table_statistics(client):
    stats_query = """
    SELECT 
      COUNT(*) AS row_count,
      COUNTIF(ARRAY_LENGTH(embeddings) = 0) AS tickets_with_empty_embeddings,
      COUNTIF(ARRAY_LENGTH(embeddings) > 0) AS tickets_with_non_empty_embeddings
    FROM `looker-tickets.zendesk.conversations_update_deduped_clean`
    """
    query_job = client.query(stats_query)
    results = query_job.result()
    operation_counter['get'] += 1
    return next(iter(results))

def verify_counts(stats):
    logging.info("Current table statistics:")
    logging.info(f"  Total rows: {stats.row_count}")
    logging.info(f"  Tickets with empty embeddings: {stats.tickets_with_empty_embeddings}")
    logging.info(f"  Tickets with non-empty embeddings: {stats.tickets_with_non_empty_embeddings}")
    
    if stats.row_count != (stats.tickets_with_empty_embeddings + stats.tickets_with_non_empty_embeddings):
        logging.warning("Count mismatch detected between total rows and embedding status")
        return False
    
    logging.info("Counts verified successfully")
    return True

def process_batches(client, dry_run=False, force=False):
    # Get table statistics
    stats = get_table_statistics(client)
    
    counts_verified = verify_counts(stats)
    if not counts_verified and not force:
        logging.error("Count mismatch detected. Use --force to proceed anyway.")
        return

    total_rows = stats.tickets_with_empty_embeddings
    logging.info(f"Total tickets to process: {total_rows}")

    last_ticket_id = load_checkpoint()
    if last_ticket_id == float('inf'):
        last_ticket_id = get_highest_ticket_id(client)
        if last_ticket_id is None:
            logging.info("No rows to process.")
            return

    total_processed = 0
    start_time = time.time()
    all_embeddings_data = []

    with tqdm(total=total_rows, desc="Overall progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]', ncols=100) as pbar:
        while total_processed < total_rows:
            query = f"""
                SELECT DISTINCT ticket_id, conversation_text, embeddings
                FROM `looker-tickets.zendesk.conversations_update_deduped_clean`
                WHERE ARRAY_LENGTH(embeddings) = 0
                AND ticket_id <= {last_ticket_id}
                ORDER BY ticket_id DESC
                LIMIT {BATCH_SIZE}
            """
            query_job = client.query(query)
            rows_to_process = list(query_job.result())
            logging.info(f"Fetched {len(rows_to_process)} distinct tickets from BigQuery")

            if not rows_to_process:
                logging.info("No more rows to process.")
                break

            embeddings_data = process_batch(rows_to_process, pbar, dry_run)

            if embeddings_data:
                all_embeddings_data.extend(embeddings_data)
                last_ticket_id = min(ticket_id for ticket_id, _ in embeddings_data) - 1
                save_checkpoint(last_ticket_id)
                
                if len(all_embeddings_data) >= UPDATE_BATCH_SIZE:
                    updated_count = update_embeddings(client, all_embeddings_data, dry_run)
                    total_processed += updated_count
                    pbar.update(updated_count)
                    all_embeddings_data = []
            else:
                last_ticket_id = min(row['ticket_id'] for row in rows_to_process) - 1
                save_checkpoint(last_ticket_id)

            pbar.set_postfix({'Processed': f"{total_processed}/{total_rows}"})

    if all_embeddings_data:
        updated_count = update_embeddings(client, all_embeddings_data, dry_run)
        total_processed += updated_count

    logging.info(f"Total tickets processed: {total_processed}")
    logging.info(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    logging.info(f"Operation summary: {dict(operation_counter)}")

def save_checkpoint(ticket_id):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(ticket_id))

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read().strip())
    return float('inf')

def process_batch(rows, pbar, dry_run=False):
    results = []
    for row in rows:
        if row['embeddings']:
            logging.warning(f"Skipping ticket_id: {row['ticket_id']} as it already has embeddings")
            continue
        embedding = get_embedding(row['conversation_text'])
        if embedding:
            results.append((row['ticket_id'], embedding))
        else:
            logging.warning(f"Failed to get embedding for ticket_id: {row['ticket_id']}")
    return results

def get_highest_ticket_id(client):
    query = """
        SELECT MAX(ticket_id) as max_id
        FROM `looker-tickets.zendesk.conversations_update_deduped_clean`
        WHERE ARRAY_LENGTH(embeddings) = 0
    """
    query_job = client.query(query)
    results = query_job.result()
    operation_counter['get'] += 1
    for row in results:
        return row.max_id
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process embeddings for BigQuery table.")
    parser.add_argument('--dry-run', action='store_true', help="Perform a dry run without updating BigQuery")
    parser.add_argument('--force', action='store_true', help="Force processing even if counts don't match expected values")
    args = parser.parse_args()

    client = bigquery.Client()

    logging.info("Starting batch processing.")
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logging.info("Deleted existing checkpoint file.")
        
        process_batches(client, dry_run=args.dry_run, force=args.force)
    except KeyboardInterrupt:
        logging.info("Script interrupted. Progress has been saved. You can resume later.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logging.info("Processing finished.")