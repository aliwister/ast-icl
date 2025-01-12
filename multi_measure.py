import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import json
import evaluate

from util.dataset import load_new_test_dataset

def calucate_sbert(model, query_sentences, concat_ref):
    scores = []
    # Loop through each query sentence and its corresponding reference sentences
    for i, query_sentence in enumerate(query_sentences):       
        variations = concat_ref[i]
        
        query_embedding = model.encode(query_sentence, convert_to_tensor=True)
        reference_embeddings = model.encode(variations, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(query_embedding, reference_embeddings)
        best_score, best_sentence_idx = torch.max(cosine_scores, dim=1)
        best_reference_sentence = variations[best_sentence_idx]

        scores.append(best_score)
    return torch.mean(torch.stack(scores)).item(), torch.std(torch.stack(scores)).item()

def compute_bleu(query_sentences, use_refs):
    bleu = evaluate.load("bleu")
    bleu_scores = []
    for pred, ref in zip(query_sentences, use_refs):
        score = bleu.compute(predictions=[pred], references=[ref])['bleu']
        bleu_scores.append(score)
    bleu_scores_tensor = torch.tensor(bleu_scores)

    # Compute mean and standard deviation
    bleu_scores_np = np.array(bleu_scores)
    mean = np.mean(bleu_scores_np)
    std = np.std(bleu_scores_np)
    
    return mean, std

def multi_measure(dataset, prediction_csv, time, limit=-1):
    df_test = load_new_test_dataset(dataset)
    use_refs = df_test['utterances'].to_list()
    if (limit > 0):
        use_refs = use_refs[:limit]

    # Load pre-trained SBERT model
    model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    model2 = SentenceTransformer('paraphrase-distilroberta-base-v1')

    file_path = 'EXPERIMENTS.txt'
    
    with open(file_path, 'a') as file:
        df = pd.read_csv(prediction_csv)['pred'] 

        if(len(df) != len(use_refs)):
            print(f"Size mismatch: {len(df)}, {len(use_refs)}")

        query_sentences = df.to_list()
        score1, std1 = calucate_sbert(model1, query_sentences, use_refs)
        score2, std2 = calucate_sbert(model2, query_sentences, use_refs)
        #bleu_score = bleu.compute(predictions=query_sentences, references=use_refs)
        blue, bstd = compute_bleu(query_sentences, use_refs)
        file.write(f"{prediction_csv}, {dataset} ,{score1}, {std1}, {score2}, {std2}, {blue}, {bstd}" + '\n') 