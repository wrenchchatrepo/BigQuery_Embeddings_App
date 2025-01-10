"""
BigQuery Embedding Script

This script processes conversation texts from a BigQuery table, generates embeddings for them,
and updates the table with the new embeddings. It uses batch processing for improved performance,
processing records in descending order of ticket_id.
"""

import os
import requests
import subprocess
from google.cloud import bigquery
import logging
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import Counter
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHECKPOINT_FILE = 'ticket_id_checkpoint.txt'
BATCH_SIZE = 500  # Number of rows to fetch from BigQuery in each query
UPDATE_BATCH_SIZE = 100  # Number of rows to accumulate before updating BigQuery
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

def update_embeddings(client, embeddings_data, dry_run=False):
    table_id = 'looker-tickets.zendesk.conversations_update'
    table = client.get_table(table_id)

    rows_to_update = []
    for ticket_id, embedding in embeddings_data:
        rows_to_update.append({
            'ticket_id': ticket_id,
            'embeddings': embedding
        })

    if dry_run:
        logging.info(f"Dry run: Would update {len(rows_to_update)} rows in BigQuery")
        return len(rows_to_update)

    errors = client.insert_rows_json(table, rows_to_update)
    if errors:
        logging.error(f"Errors occurred while inserting rows: {errors}")
        return 0
    else:
        logging.info(f"BigQuery insert_rows_json() completed without errors for {len(rows_to_update)} rows")

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

    return updated_count
addPreferredContent
"""
BigQuery Embedding Script

This script processes conversation texts from a BigQuery table, generates embeddings for them,
and updates the table with the new embeddings. It uses batch processing for improved performance,
processing records in descending order of ticket_id.
"""

import os
import requests
import subprocess
from google.cloud import bigquery
import logging
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import Counter
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHECKPOINT_FILE = 'ticket_id_checkpoint.txt'
BATCH_SIZE = 500  # Number of rows to fetch from BigQuery in each query
UPDATE_BATCH_SIZE = 100  # Number of rows to accumulate before updating BigQuery
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

def update_embeddings(client, embeddings_data, dry_run=False):
    table_id = 'looker-tickets.zendesk.conversations_update'
    table = client.get_table(table_id)

    rows_to_update = []
    for ticket_id, embedding in embeddings_data:
        rows_to_update.append({
            'ticket_id': ticket_id,
            'embeddings': embedding
        })

    if dry_run:
        logging.info(f"Dry run: Would update {len(rows_to_update)} rows in BigQuery")
        return len(rows_to_update)

    errors = client.insert_rows_json(table, rows_to_update)
    if errors:
        logging.error(f"Errors occurred while inserting rows: {errors}")
        return 0
    else:
        logging.info(f"BigQuery insert_rows_json() completed without errors for {len(rows_to_update)} rows")

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

    return updated_count

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
        FROM `project_id.dataset_1.table_1`
        WHERE ARRAY_LENGTH(embeddings) = 0
    """
    query_job = client.query(query)
    results = query_job.result()
    operation_counter['get'] += 1
    for row in results:
        return row.max_id
    return None

def get_table_statistics(client):
    stats_query = """
        WITH ticket_stats AS (
            SELECT
                ticket_id,
                CASE WHEN ARRAY_LENGTH(embeddings) > 0 THEN 1 ELSE 0 END as has_embedding
            FROM `project_id.dataset_1.table_1`
        )
        SELECT 
            COUNT(DISTINCT ticket_id) as total_distinct_tickets,
            COUNTIF(has_embedding = 1) as total_records_with_embeddings,
            COUNT(DISTINCT CASE WHEN has_embedding = 1 THEN ticket_id END) as distinct_tickets_with_embeddings,
            COUNTIF(has_embedding = 0) as total_records_without_embeddings,
            COUNT(DISTINCT CASE WHEN has_embedding = 0 THEN ticket_id END) as distinct_tickets_without_embeddings
        FROM ticket_stats
    """
    query_job = client.query(stats_query)
    results = query_job.result()
    operation_counter['get'] += 1
    return next(iter(results))

def verify_counts(stats):
    total_distinct = stats.total_distinct_tickets
    with_embeddings_distinct = stats.distinct_tickets_with_embeddings
    without_embeddings_distinct = stats.distinct_tickets_without_embeddings
    
    logging.info("Current table statistics:")
    logging.info(f"  Total distinct tickets: {total_distinct}")
    logging.info(f"  Distinct tickets with embeddings: {with_embeddings_distinct}")
    logging.info(f"  Distinct tickets without embeddings: {without_embeddings_distinct}")
    
    if total_distinct != (with_embeddings_distinct + without_embeddings_distinct):
        logging.warning("Count mismatch detected:")
        logging.warning(f"  Total distinct tickets ({total_distinct}) does not equal")
        logging.warning(f"  sum of tickets with ({with_embeddings_distinct}) and without ({without_embeddings_distinct}) embeddings")
        return False
    else:
        logging.info("Counts verified successfully")
        return True

def process_batches(client, dry_run=False, force=False):
    # Get table statistics
    stats = get_table_statistics(client)
    logging.info(f"Table statistics:")
    logging.info(f"  Total distinct tickets: {stats.total_distinct_tickets}")
    logging.info(f"  Total records with embeddings: {stats.total_records_with_embeddings}")
    logging.info(f"  Distinct tickets with embeddings: {stats.distinct_tickets_with_embeddings}")
    logging.info(f"  Total records without embeddings: {stats.total_records_without_embeddings}")
    logging.info(f"  Distinct tickets without embeddings: {stats.distinct_tickets_without_embeddings}")
    
    counts_verified = verify_counts(stats)
    if not counts_verified and not force:
        logging.error("Count mismatch detected. Use --force to proceed anyway.")
        return

    total_rows = stats.distinct_tickets_without_embeddings
    logging.info(f"Total distinct tickets to process: {total_rows}")

    last_ticket_id = load_checkpoint()
    if last_ticket_id == float('inf'):
        last_ticket_id = get_highest_ticket_id(client)
        if last_ticket_id is None:
            logging.info("No rows to process.")
            return

    total_processed = 0
    start_time = time.time()
    all_embeddings_data = []

    # Initialize the progress bar with the total number of rows to process
    with tqdm(total=total_rows, desc="Overall progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]', ncols=100) as pbar:
        while total_processed < total_rows:
            query = f"""
                SELECT DISTINCT ticket_id, conversation_text, embeddings
                FROM `looker-tickets.zendesk.conversations_update`
                WHERE ARRAY_LENGTH(embeddings) = 0
                AND ticket_id <= {last_ticket_id}
                ORDER BY ticket_id DESC
                LIMIT {BATCH_SIZE}
            """
            query_start = time.time()
            query_job = client.query(query)
            rows_to_process = list(query_job.result())
            query_time = time.time() - query_start
            logging.info(f"Fetched {len(rows_to_process)} distinct tickets from BigQuery in {query_time:.2f} seconds")
            operation_counter['get'] += 1

            if not rows_to_process:
                logging.info("No more rows to process.")
                break

            logging.info(f"Processing batch of {len(rows_to_process)} distinct tickets")
            process_start = time.time()
            embeddings_data = process_batch(rows_to_process, pbar, dry_run)
            process_time = time.time() - process_start
            logging.info(f"Processed {len(embeddings_data)} embeddings in {process_time:.2f} seconds")

            if embeddings_data:
                all_embeddings_data.extend(embeddings_data)
                last_ticket_id = min(ticket_id for ticket_id, _ in embeddings_data) - 1
                save_checkpoint(last_ticket_id)
                
                # Update BigQuery if we've accumulated enough rows
                if len(all_embeddings_data) >= UPDATE_BATCH_SIZE:
                    update_start = time.time()
                    updated_count = update_embeddings(client, all_embeddings_data, dry_run)
                    update_time = time.time() - update_start
                    logging.info(f"{'Dry run: Would update' if dry_run else 'Updated'} {updated_count} rows in BigQuery in {update_time:.2f} seconds")
                    total_processed += updated_count
                    pbar.update(updated_count)
                    all_embeddings_data = []
            else:
                logging.warning(f"No valid embeddings in this batch. Moving to next batch.")
                last_ticket_id = min(row['ticket_id'] for row in rows_to_process) - 1
                save_checkpoint(last_ticket_id)

            pbar.set_postfix({'Processed': f"{total_processed}/{total_rows}"})
            time.sleep(1)  # Small delay to avoid overwhelming the API

            # Update total_rows count
            new_stats = get_table_statistics(client)
            new_total_rows = new_stats.distinct_tickets_without_embeddings
            if new_total_rows != total_rows:
                diff = new_total_rows - total_rows
                total_rows = new_total_rows
                pbar.total = total_rows
                logging.info(f"Updated total distinct tickets to process. New total: {total_rows} (Change: {diff:+d})")

    # Update any remaining embeddings
    if all_embeddings_data:
        updated_count = update_embeddings(client, all_embeddings_data, dry_run)
        total_processed += updated_count

    logging.info(f"Total distinct tickets processed: {total_processed}")
    logging.info(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    logging.info(f"Operation summary: {dict(operation_counter)}")

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
