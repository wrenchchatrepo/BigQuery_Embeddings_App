# README_Zendesk_Ticket.md

## Summary:

This system allows for comprehensive analysis of support tickets, including automatic categorization of issues and resolutions, evaluation of agent performance, and prediction of customer satisfaction. It combines structured ticket data with natural language processing techniques (embeddings) to gain deeper insights into customer support interactions.

## Architecture:

```
Project: looker-tickets
  Dataset: zendesk
    Tables:
	  csat_boosted_tree_classifier
	  cosine_similarity
	  AvaKeyWords
	  confidence_score
	  conversations_complete
	  vector_similarity_scores
	  tickets_complete
```

## Data Collection and Storage:

+ The core of this system is the `tickets_complete` table. This table contains detailed information about each support ticket, including customer details, agent information, ticket status, and various metrics.
+ The conversations_complete table stores the actual content of the conversations related to each ticket, including the embeddings (vector representations) of the conversation text.

## Issue and Resolution Classification:

+ The `AvaKeyWords` table acts as a reference for predefined issues and resolutions. It contains embeddings for each issue and resolution type.
+ The `vector_similarity_scores` table is created by comparing the embeddings from conversations_complete with those in `AvaKeyWords`. It stores the similarity scores between each ticket's conversation and potential issues/resolutions.
+ The `cosine_similarity` routine is used in this process to calculate the similarity between embeddings.

## Best Match Assignment:

The results from `vector_similarity_scores` are used to assign the best-matched issue and resolution to each ticket in the `tickets_complete` table (columns: Best_Matched_Issue, Best_Matched_Resolution, Best_Issue_Similarity, Best_Resolution_Similarity).

## Agent Performance Analysis:

The `confidence_score` table provides an overview of agent performance, including the number of tickets handled, average sentiment of their tickets, and a calculated confidence score.

## Customer Satisfaction Prediction:

+ The `csat_boosted_tree_classifier` is a machine learning model in the system.
+ It uses the following features from the `tickets_complete` table:
	- Is_Escalated
	- Is_FRT_Slo_Breach
	- Is_Autosolved
+ The model predicts a predicted_csat_category, which is an INT64 value representing different levels of customer satisfaction.

## Process Flow:

1. Ticket data is collected and stored in `tickets_complete`.

2. Conversation data is processed and stored in `conversations_complete`, including the generation of embeddings.

3. The `cosine_similarity` routine is used to compare conversation embeddings with predefined issues and resolutions from AvaKeyWords.

4. Similarity scores are stored in `vector_similarity_scores`.

5. Best matches for issues and resolutions are assigned to each ticket in `tickets_complete`.

6. Agent performance metrics are calculated and stored in `confidence_score`.

7. The `csat_boosted_tree_classifier` model uses ticket data to predict customer satisfaction.

