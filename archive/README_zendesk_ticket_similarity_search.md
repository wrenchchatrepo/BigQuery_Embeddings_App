# Zendesk Ticket Similarity Search

This project implements a similarity search for Zendesk tickets using embeddings and BigQuery.<br>It allows you to find tickets that are semantically similar to a given input text.

## Overview

The script performs the following tasks:
1. Generates an embedding for an input text using OpenAI's API.
2. Executes a BigQuery job to calculate similarity scores between the input embedding and all ticket embeddings in the database.
3. Stores the results in a BigQuery table.
4. Retrieves and displays the most similar tickets based on a configurable similarity threshold.

## Prerequisites

- Python 3.7+
- Google Cloud SDK
- BigQuery setup with appropriate permissions
- OpenAI API key (or access to a local LM Studio server)

## Setup

1. Clone this repository:
```
git clone <repository-url>
cd zendesk-similarity-search
```
2. Install the required Python packages:
```   
pip install openai google-cloud-bigquery
```
3. Set up your Google Cloud credentials:
```
gcloud auth application-default login
```
4. Configure the OpenAI API key or LM Studio server details in the script.

## Usage

The script operates in two modes:

1. Start a new similarity search job:
```   
python similarity_search.py
```
This generates the embedding for the input text and starts the BigQuery job to calculate similarities.

2. Check the results of a previously started job:
```
python similarity_search.py check
```
This retrieves and displays the results of the similarity search.

## Configuration

- Modify the SIMILARITY_THRESHOLD constant in the script to adjust the minimum similarity score for matching tickets.
- Update the conversation_text variable in the start_job() function to change the input text for similarity search.

## Output

The script outputs:
- Total number of matching tickets
- Average, minimum, and maximum similarity scores
- A list of all matching tickets with their IDs and similarity scores

## BigQuery Schema

The script assumes the following BigQuery table structure:

1. looker-tickets.zendesk.conversations_complete:
   - ticket_id: INTEGER
   - embeddings: ARRAY<FLOAT64>

2. looker-tickets.zendesk.similarity_search_results (created by the script):
   - ticket_id: INTEGER
   - similarity_score: FLOAT64

## Troubleshooting

- If the script hangs or takes too long, try running the "check" command multiple times.
- Ensure your BigQuery permissions are correctly set up.
- Verify that the input text is relevant to your ticket database for meaningful results.

## Future Improvements

- Implement error handling for BigQuery job failures.
- Add support for batch processing of multiple input texts.
- Create a user interface for easier interaction with the similarity search tool.

