# Zendesk Ticket Embedding and Similarity Search

## Overview

This application provides a powerful solution for analyzing and searching Zendesk support tickets using advanced natural language processing techniques. By generating vector embeddings for ticket content and storing them in Google BigQuery, the system enables semantic similarity searches that go beyond simple keyword matching.

### Key Features
- Generate embeddings for Zendesk ticket content using local LLM (Language Model)
- Store and manage embeddings efficiently in BigQuery
- Perform semantic similarity searches across ticket database
- Support both batch processing and real-time embedding generation
- Flexible deployment options (local development or cloud function)

### Why This Matters
Support teams often need to find similar past tickets to:
- Identify recurring issues
- Apply previously successful solutions
- Understand patterns in customer inquiries
- Improve response consistency and efficiency

Traditional keyword-based searches can miss contextually similar tickets. This system uses embeddings to capture the semantic meaning of ticket content, enabling more intelligent and relevant search results.

## Project Structure

```
BQ_Embeddings/
├── .env                        # Environment variables configuration
├── .gitignore                 # Git ignore file
├── bq_embedding.py            # Batch processing for BigQuery ticket data
├── embed_cloud_fn.py          # Cloud Function for embedding generation
├── get_embedding.py           # Core embedding generation utility
├── get_embeddings_local_llm.py # Local LLM integration for embeddings
├── main.py                    # Flask application for local development
├── similarity_search.py       # Similarity search implementation
├── looker-tickets_zendesk_schema.md # BigQuery schema documentation
└── archive/                   # Archived documentation and legacy code
```

### Core Components

1. **Embedding Generation**
   - `get_embedding.py`: Core utility for generating embeddings
   - `embed_cloud_fn.py`: Cloud Function implementation
   - `main.py`: Local development server
   - Uses LM Studio with All-MiniLM-L6-v2-Embedding-GGUF model

2. **Data Processing**
   - `bq_embedding.py`: Batch processes tickets in BigQuery
   - `get_embeddings_local_llm.py`: Local LLM integration
   - Handles missing embeddings and updates

3. **Similarity Search**
   - `similarity_search.py`: Implements cosine similarity search
   - Uses BigQuery for efficient vector comparisons
   - Configurable similarity thresholds

### BigQuery Architecture

The system uses the following BigQuery structure:
```
Project: looker-tickets
└── Dataset: zendesk
    ├── conversations_complete    # Stores ticket content and embeddings
    ├── similarity_search_results # Temporary results storage
    └── Additional tables for analytics and processing
```

## Prerequisites

1. **Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install openai google-cloud-bigquery flask requests
   ```

2. **LM Studio**
   - Download and install LM Studio from their official website
   - Load the "All-MiniLM-L6-v2-Embedding-GGUF" model
   - Start the local server (default port: 1234)

3. **ngrok**
   - Install ngrok
   - Set up tunnel to LM Studio:
     ```bash
     ngrok http 1234
     ```
   - Note the generated URL (e.g., https://your-tunnel.ngrok.io)

4. **Google Cloud Setup**
   - Install Google Cloud SDK
   - Authenticate:
     ```bash
     gcloud auth application-default login
     ```
   - Set your project:
     ```bash
     gcloud config set project your-project-id
     ```

## BigQuery Setup

1. **Create Dataset**
   ```sql
   CREATE SCHEMA IF NOT EXISTS `your-project.zendesk`;
   ```

2. **Create Tables**
   ```sql
   -- Table for storing conversations with embeddings
   CREATE TABLE IF NOT EXISTS `your-project.zendesk.conversations_complete` (
     ticket_id INT64,
     embeddings ARRAY<FLOAT64>
   );

   -- Table for similarity search results (created automatically by the script)
   -- your-project.zendesk.similarity_search_results
   ```

## Configuration

1. **Environment Variables**
   Create a .env file:
   ```
   GOOGLE_CLOUD_PROJECT=your-project-id
   LM_STUDIO_URL=http://localhost:1234
   NGROK_URL=your-ngrok-url
   ```

## Running the Application

### 1. Generate Embeddings

1. **Start Local Services**
   ```bash
   # Start LM Studio and ensure it's running on port 1234
   # Start ngrok tunnel
   ngrok http 1234
   ```

2. **Update URLs**
   - Copy your ngrok URL
   - Update `embed_cloud_fn.py` and `main.py` with the new URL

3. **Run Embedding Generation**
   ```bash
   # For local development
   python main.py
   
   # For cloud function (if deployed)
   curl -X POST https://your-cloud-function-url/get-embedding \
     -H "Content-Type: application/json" \
     -d '{"text": "your text here"}'
   ```

### 2. Execute Similarity Search

1. **Start a New Search**
   ```bash
   python similarity_search.py
   ```
   This will:
   - Generate embedding for your input text
   - Create/update the similarity_search_results table
   - Start the comparison job

2. **Check Results**
   ```bash
   python similarity_search.py check
   ```
   This will show:
   - Total matching tickets
   - Average similarity score
   - Minimum and maximum scores
   - List of all matching tickets

## Troubleshooting

1. **LM Studio Connection Issues**
   - Verify LM Studio is running (http://localhost:1234)
   - Check ngrok tunnel status
   - Verify URLs in configuration

2. **BigQuery Issues**
   - Verify Google Cloud authentication
   - Check project and dataset permissions
   - Validate table schemas

3. **Common Error Messages**
   - "Failed to get embeddings": Check LM Studio and ngrok connection
   - "Permission denied": Verify Google Cloud credentials
   - "Table not found": Ensure BigQuery tables are created

## Monitoring

1. **Check Logs**
   ```bash
   # For cloud function
   gcloud functions logs read

   # For local server
   Check terminal output
   ```

2. **Monitor BigQuery Usage**
   - Visit Google Cloud Console > BigQuery
   - Check query history and job status

## Best Practices

1. **Performance**
   - Use batch processing for large datasets
   - Monitor memory usage when generating multiple embeddings
   - Consider implementing rate limiting for API endpoints

2. **Security**
   - Keep credentials secure
   - Use environment variables for sensitive data
   - Regularly rotate API keys

3. **Maintenance**
   - Regularly update dependencies
   - Monitor ngrok tunnel stability
   - Back up important data and configurations
