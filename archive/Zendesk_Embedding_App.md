# README: Zendesk Embedding App Setup and Execution Guide

Llama 3.2 Local Embedding Generation via LM Studio and Cloud Function

This project uses the local llama3.1 model in LM Studio, coupled with a Google Cloud Function to generate text embeddings. The embedding generation is integrated with Google BigQuery for large-scale processing.

## Prerequisites

1. Install LM Studio with support for llama3.1.

2. Google Cloud SDK installed and authenticated.

3. gcloud CLI configured with access to the appropriate GCP project.

4. A BigQuery table looker-tickets.zendesk.AvaKeyWords with columns for storing text and embeddings.

## Architecture Explanation

This project uses multiple tools to process Zendesk ticket conversation text and generate embeddings, which are then stored in BigQuery. Below is an overview of the architecture:

1. Virtual Environment Setup: Python virtual environment to manage dependencies and run scripts.

2. Embedding Generation: Embeddings are generated via LM Studio, which is exposed through an ngrok tunnel to provide a publicly accessible endpoint.

3. BigQuery Storage: Zendesk ticket data is stored in BigQuery. The embedding generation process retrieves conversation text, generates embeddings, and updates BigQuery with the embeddings.

4. LM Studio and Ngrok: LM Studio is used to generate embeddings locally. Ngrok is used to expose LM Studio's local endpoint to the internet for easy access.

## Project Structure

The following files are included in this project:

**requirements.txt**: Contains the dependencies needed for the Cloud Function.

**get_embedding.py**: Defines the embedding generation logic, calling the local llama3.1 model via LM Studio.

**embed_cloud_fn.py**: Contains code for deploying and managing the Cloud Function.

**bq_embedding.py**: A Python script for querying BigQuery, retrieving rows without embeddings, and updating them by calling the Cloud Function.

**.gcloudignore.md**: Files and directories to be excluded during the Cloud Function deployment process.

## Instructions

### Step 1: Install Python Virtual Environment and Dependencies

0. Navigate to the correct directory:  

```
z /Users/dionedge/dev/doit/zd_tickets
```

1. Create and activate a virtual environment:

```
python3 -m venv myenv
source myenv/bin/activate
```

2.  Install the required dependencies:

```
pip install -r requirements.txt
```

### Step 2: Configure LM Studio with Llama 3.1

1. Download LM Studio from: https://releases.lmstudio.ai/darwin/arm64/0.3.3/LM-Studio-0.3.3-arm64.dmg

2. Example models: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf and second-state/All-MiniLM-L6-v2-Embedding-GGUF

3. Embeddings endpoint: http://localhost:1234/v1/embeddings

4. LM Studio Configuration:
+ GPU Offload: max
+ Context Length: 131072
+ Temperature: 0.5
+ Tokens to Generate: -1
+ CPU Thread: 20
+ Keep Entire Model in RAM: TRUE
+ Config File: lmstudio_config.json

### Step 3: Cloud Function Setup

0.  Authenticate with Google Cloud:
```
gcloud auth login
gcloud config set project looker-tickets
gcloud components update
```

1. Navigate to the correct directory:  

```
z /Users/dionedge/dev/doit/zd_tickets
```

2. Update IAM Policy to allow your account to invoke the function:

```
gcloud functions add-iam-policy-binding get_embedding --region=us-central1 --member=user:dion@wrench.chat --role=roles/cloudfunctions.invoker
```

3. Deploy the Cloud Function using the embed_cloud_fn.py script.

```
python3 embed_cloud_fn.py
```

4. This will deploy a Google Cloud Function that calls your local LM Studio setup for embedding generation. Export your authentication token:

```
export TOKEN=$(gcloud auth print-identity-token)
```

5. Deploy the function:

```
gcloud functions deploy get_embedding –runtime python39 –trigger-http –allow-unauthenticated –entry-point get_embedding
```

### Step 5: Ngrok Setup

1. Install ngrok (once)

```
brew install ngrok/ngrok/ngrok
```

2. Go to ngrok dashboard (https://dashboard.ngrok.com/get-started/setup/macos) and copy the ngrok config add-authtoken command, then run it in your terminal:

```
ngrok config add-authtoken $YOUR_AUTHTOKEN
```

3. Start ngrok to expose LM Studio:

```
ngrok http --url=ngrok.wrench.chat 1234
```

4. Domain: https://ngrok.wrench.chat

+ TLS Certificate is automatically provisioned and renewed for this domain.

### Step 6: Test

Test the embeddings endpoint:

```
curl -X POST https://ngrok.wrench.chat/v1/embeddings -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"input": ["This is a test"]}'
```

### Step 7. Running the BigQuery Embedding Script

1. Obtain a Google Cloud identity token and activate your venv

```
export TOKEN=$(gcloud auth print-identity-token)
source myenv/bin/activate
gcloud auth application-default login
```

2. Run the main embedding script

```
python3 bq_embedding.py
```

3. Query BigQuery for Validation

```
SELECT COUNT(*) FROM looker-tickets.zendesk.conversations WHERE ARRAY_LENGTH(embeddings) = 0;
```

## Overview of Files

+ bq_embedding.py: The main script responsible for driving the embedding process. It interacts with BigQuery to retrieve conversation data and updates the rows with newly generated embeddings from LM Studio.

+ get_embedding.py: A script that handles the request for generating embeddings from LM Studio (accessed via ngrok) and returns the result to be processed by bq_embedding.py.

+ embed_cloud_fn.py: A script that defines the Cloud Function behavior, which can optionally be used to generate embeddings in a cloud-based setup, as opposed to local execution through LM Studio.

+ requirements.txt: Contains a list of necessary Python dependencies such as google-cloud-bigquery and requests to ensure all scripts can run properly.

+ .gcloudignore.md: Specifies the files and directories to be excluded when deploying the Cloud Function to avoid uploading unnecessary files or large datasets.

+ main.py: The entry point for the Cloud Function. This script receives text input, makes a request to LM Studio (via ngrok), and returns the embedding response. It’s responsible for handling the HTTP requests and responses as part of the Cloud Function deployment.

## LM Studio Parameters

+ Batch Size Optimization: Increase batch size to 2000-5000 and implement dynamic adjustment.

+ Parallelization: Increase worker threads to 20-30 and dynamically adjust based on CPU load.

+ Query for Null Embeddings: Modify query to target ARRAY_LENGTH(embeddings) = 0.

+ Retry Mechanism with Exponential Backoff: Implement retries for Cloud Function failures with exponential backoff.

+ Batch Updates for BigQuery: Group updates into batches of 500-1000 rows for more efficient writes.

+ Reduce API Overhead: Process multiple texts per API call (10-20 per request).

+ Optimize LM Studio: Increase token limit and optimize memory allocation for handling large conversation texts.

+ Retry for BigQuery Writes: Implement exponential backoff for failed BigQuery updates.

+ Asynchronous I/O: Implement asynchronous fetching and writing for BigQuery and Cloud Functions.

+ System Monitoring: Monitor system resources and dynamically adjust batch size, threads, and query limits.

+ Avoid Overwhelming BigQuery: Implement rate limiting and throttling based on API quotas and system load.