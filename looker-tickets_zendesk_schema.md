looker-tickets_zendesk_schema

+ Model
  - looker-tickets.zendesk.csat_boosted_tree_classifier
  - Labels: predicted_csat_category INT64
  - Features: looker-tickets.zendesk.new_tickets
	> Priority STRING<br>
	> Is_Escalated BOOL<br>
	> Is_FRT_Slo_Breach BOOL<br>
	> Is_Autosolved BOOL<br>
	> Is_Cloud_Support_Case_Used BOOL<br>
	> Cloud_Platform STRING<br>
	> Product STRING<br>
	> Agent_Time_Zone STRING<br>
	> Requester_Time_Zone STRING<br>
	> Requester_Region STRING<br>
	> Average_Solve_Time_Hours FLOAT64<br>
	> Average_Customer_Mrr FLOAT64

+ Routines
  - looker-tickets.zendesk.cosine_similarity

+ Tables

  - looker-tickets.zendesk.AvaKeyWords
    > Type STRING<br>
    > Name STRING<br>
    > Definition STRING<br>
    > Keywords STRING<br>
    > Content STRING<br>
    > Embedding FLOAT<br>
    > array_length INTEGER<br>

  - looker-tickets.zendesk.confidence_score
    > agent_name STRING<br>
    > tickets INTEGER<br>
    > avg_sentiment_integer FLOAT<br>
    > confidence_score FLOAT<br>

  - looker-tickets.zendesk.conversations_complete
	> ticket_id INTEGER<br>
	> array_length INTEGER<br>
 	> embeddings FLOAT<br>

  - looker-tickets.zendesk.vector_similarity_scores
	> ticket_id INTEGER<br>
	> Name STRING<br>
	> Type STRING [Issue, Resolution]<br>
	> similarity FLOAT<br>

  - looker-tickets.zendesk.tickets_complete
	> Ticket_ID INTEGER<br>
	> Created_Date DATE<br>
	> Priority STRING<br>
	> Status STRING<br>
	> Is_Escalated BOOLEAN<br>
	> Is_Autosolved BOOLEAN<br>
	> Is_Requester_From_Doit BOOLEAN<br>
	> Is_Getcre BOOLEAN<br>
	> Is_Public BOOLEAN<br>
	> Is_FRT_Slo_Breach BOOLEAN<br>
	> Is_Cloud_Support_Case_Used BOOLEAN<br>
	> Cloud_Platform STRING<br>
	> Practice_Area STRING<br>
	> Skill_Group STRING<br>
	> Product STRING<br>
	> Subject STRING<br>
	> Agent_User_ID FLOAT<br>
	> Agent_Name STRING<br>
	> Agent_Pod STRING<br>
	> Agent_Region STRING<br>
	> Agent_Time_Zone STRING<br>
	> Requester_ID INTEGER<br>
	> Requester_Name STRING<br>
	> Requester_Email STRING<br>
	> Requester_Organization_Name STRING<br>
	> Requester_Region STRING<br>
	> Requester_Time_Zone STRING<br>
	> Rating STRING<br>
	> Rating_Comment STRING<br>
	> Rating_Comment_Length FLOAT<br>
	> Rating_Offered INTEGER<br>
	> Agent_Score INTEGER<br>
	> RND_T FLOAT<br>
	> Sentiment FLOAT<br>
	> Sentiment_Heuristic STRING<br>
	> Sentiment_Integer FLOAT<br>
	> Average_Solve_Time_Hours FLOAT<br>
	> Average_FRT_Hours FLOAT<br>
	> Average_Customer_Mrr FLOAT<br>
	> Count INTEGER<br>
	> total_comment_length INTEGER<br>
	> comment_count INTEGER<br>
	> conversation_duration_days INTEGER<br>
	> embedding_length INTEGER<br>
	> Best_Matched_Issue STRING<br>
	> Best_Matched_Resolution STRING<br>
	> Best_Issue_Similarity FLOAT<br>
	> Best_Resolution_Similarity FLOAT<br>
