import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import json


catalog_df = pd.read_csv(r"C:\Users\Shubrat Verma\OneDrive\Desktop\Speedrun101\Speedrun101\SHL_catalog.csv")
def combine_row(row):
    parts = [
        str(row["Assessment Name"]),
        str(row["Duration"]),
        str(row["Remote Testing Support"]),
        str(row["Adaptive/IRT"]),
        str(row["Test Type"]),
        str(row["Skills"]),
        str(row["Description"]),
    ]
    return ' '.join(parts)


catalog_df['combined'] = catalog_df.apply(combine_row,axis=1)
# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

#converting each row into a vector/word embedding
corpus = catalog_df['combined'].tolist()
corpus_embeddings = model.encode(corpus,convert_to_tensor=True)
corpus_embeddings
def compute_metrics(benchmark_queries,k=5):
    recall_scores = []
    average_precisions = []

    for entry in benchmark_queries:
        query = entry["query"]
        relevant_items = entry["relevant"]

        results = find_assessments(query)
        topk = [res["Assessment Name"] for res in results[:k]]

        #recall@k
        count = 0
        for item in topk:
            if item in relevant_items:
                count+=1
        recall_score = count/len(relevant_items)
        recall_scores.append(recall_score)

        #map@k
        ap = 0.0
        relevant_count = 0
        for i,res in enumerate(topk):
            if res in relevant_items:
                relevant_count+=1
                precision_at_k = relevant_count/(i+1)
                ap += precision_at_k
        ap = ap/min(k,len(relevant_items))
        average_precisions.append(ap)
        
    recall = sum(recall_scores)/len(recall_scores)
    map_ = sum(average_precisions)/len(average_precisions)
    
    print(f"Recall@{k}: {recall:.4f}")
    print(f"MAP@{k}: {map_:.4f}")



def find_assessments(user_query,k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(user_query, convert_to_tensor = True)
    cosine_scores = util.cos_sim(query_embedding,corpus_embeddings)[0]
    top_k = min(k,len(corpus))
    top_results = torch.topk(cosine_scores,k=top_k)
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()
        result = {
            "Assessment Name": catalog_df.iloc[idx]['Assessment Name'],
            "Skills": catalog_df.iloc[idx]['Skills'],
            "Test Type": catalog_df.iloc[idx]['Test Type'],
            "Description": catalog_df.iloc[idx]['Description'],
            "Remote Testing Support": catalog_df.iloc[idx]['Remote Testing Support'],
            "Adaptive/IRT": catalog_df.iloc[idx]['Adaptive/IRT'],
            "Duration": catalog_df.iloc[idx]['Duration'],
            "URL": catalog_df.iloc[idx]['URL'],
            "Score": round(score.item(), 4)
        }
        results.append(result)
    return results


benchmark_queries = [
    {
        "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes.",
        "relevant": ["Java Developer Assessment #1","Quick Java Screening #27","Java Coding Drill #47"]
    },
    {
        "query": "Suggest an assessment for a fresher data analyst that includes Python and SQL skills in under 50 minutes.",
        "relevant": ["Data Analyst Screening #3","SQL & Reporting Challenge #18","Cross-functional Assessment #41"]
    },
    {
        "query": "Looking for remote-enabled JavaScript technical assessment. Needs to be adaptive.",
        "relevant": ["JavaScript Screening Test #11"]
    },
    {
        "query": "Want to assess communication and teamwork skills in under 30 minutes.",
        "relevant": ["Communication Skills Test #19","Business Communication Evaluation #8","Communication & Team Fit #24","Interpersonal Skills Assessment #43"]
    },
]

compute_metrics(benchmark_queries,k=5)