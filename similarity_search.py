from openai import OpenAI
from google.cloud import bigquery
import time
import sys

# LM Studio client setup
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_embedding(text, model="second-state/All-MiniLM-L6-v2-Embedding-GGUF"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# BigQuery client setup
print("Setting up BigQuery client...")
bq_client = bigquery.Client()
print("BigQuery client set up.")

SIMILARITY_THRESHOLD = 0.60

def start_job():
    # Your conversation text
    conversation_text = """A company is facing issues with the Looker scheduled report integration to Google Drive and Sheets. [... rest of the text ...]"""

    print("Generating embedding...")
    embedding = get_embedding(conversation_text)
    embedding_str = ','.join(map(str, embedding))
    print("Embedding generated.")

    # BigQuery query
    main_query = f"""
    CREATE OR REPLACE TABLE `looker-tickets.zendesk.similarity_search_results` AS
    WITH new_embedding AS (
      SELECT ARRAY<FLOAT64>[{embedding_str}] AS embedding
    ),
    cosine_similarity AS (
      SELECT
        cc.ticket_id,
        (SUM(a * b) / (SQRT(SUM(a * a)) * SQRT(SUM(b * b)))) AS similarity_score
      FROM
        `looker-tickets.zendesk.conversations_complete` cc,
        new_embedding,
        UNNEST(cc.embeddings) a WITH OFFSET pos1,
        UNNEST(new_embedding.embedding) b WITH OFFSET pos2
      WHERE
        pos1 = pos2
      GROUP BY
        cc.ticket_id
    )
    SELECT 
      ticket_id,
      similarity_score
    FROM 
      cosine_similarity
    ORDER BY
      similarity_score DESC
    """

    print("Executing BigQuery job...")
    query_job = bq_client.query(main_query)
    print("Query job started. You can check the results using:")
    print("python similarity_search.py check")

def check_results():
    try:
        # Check if the results table exists
        table_ref = bq_client.dataset("zendesk").table("similarity_search_results")
        bq_client.get_table(table_ref)
        
        print(f"Results table found. Fetching results with similarity > {SIMILARITY_THRESHOLD}...")
        results_query = f"""
        SELECT 
          ticket_id,
          similarity_score
        FROM `looker-tickets.zendesk.similarity_search_results`
        WHERE similarity_score > {SIMILARITY_THRESHOLD}
        ORDER BY similarity_score DESC
        """
        results_job = bq_client.query(results_query)

        results = list(results_job.result())
        
        if results:
            print(f"Total matches: {len(results)}")
            print(f"Average similarity score: {sum(row.similarity_score for row in results) / len(results):.4f}")
            print(f"Minimum similarity score: {min(row.similarity_score for row in results):.4f}")
            print(f"Maximum similarity score: {max(row.similarity_score for row in results):.4f}")

            print("\nAll matching tickets:")
            for row in results:
                print(f"Ticket ID: {row.ticket_id}, Similarity Score: {row.similarity_score:.4f}")
        else:
            print(f"No tickets found with similarity score > {SIMILARITY_THRESHOLD}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("The results may not be ready yet. Please try again later.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_results()
    else:
        start_job()

print("Script completed.")
