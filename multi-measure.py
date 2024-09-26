import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import json
import torch.nn.functional as F
import evaluate

from util.dataset import load_dataset

def calucate_sbert(model, query_sentences, concat_ref):
    scores = []
    # Loop through each query sentence and its corresponding reference sentences
    for i, query_sentence in enumerate(query_sentences):
        #print(f"Processing query: '{query_sentence}'")
        
        variations = concat_ref[i]
        
        # Encode the query and reference sentences
        query_embedding = model.encode(query_sentence, convert_to_tensor=True)
        reference_embeddings = model.encode(variations, convert_to_tensor=True)
        
        # Calculate cosine similarity scores
        cosine_scores = util.cos_sim(query_embedding, reference_embeddings)
        
        # Get the best score and corresponding reference sentence
        best_score, best_sentence_idx = torch.max(cosine_scores, dim=1)
        best_reference_sentence = variations[best_sentence_idx]

        scores.append(best_score)
    return torch.mean(torch.stack(scores)).item()

# Load pre-trained SBERT model
model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model2 = SentenceTransformer('paraphrase-distilroberta-base-v1')

files =  [
#prompt output files
]

def load_refs(dataset):
    _, df_test = load_dataset(dataset)
    mult_ref_json = f"new_dataset/output.{dataset}.test.json"
    with open(mult_ref_json, 'r') as file:
        data = json.load(file)
    new_refs = [x['final'] for x in data]
    original_ref = df_test['utterance'].values[:293]
    final_ref = np.concatenate((new_refs, original_ref.reshape(-1, 1)), axis=1)
    return final_ref

cosql_refs = load_refs('cosql')
sparc_refs = load_refs('sparc')
spider_refs = load_refs('spider')

file_path = 'MULTI_EXPERIMENTS_FINAL.txt'
bleu = evaluate.load("bleu")
with open(file_path, 'a') as file:
    for obj in files:
        print(obj['file'])
        prediction_csv = obj['file']
        df = pd.read_csv(prediction_csv)[:150] 
        if (obj['dataset'] == "cosql"):
            use_refs = cosql_refs
        elif(obj['dataset'] == "spider"):
            use_refs = spider_refs
        elif (obj['dataset'] == "sparc"):
            use_refs = sparc_refs

        if(len(df) != len(use_refs)):
            print(f"Size mismatch: {len(df)}, {len(use_refs)}")
            continue  
        query_sentences = [item for sublist in df.values for item in sublist]                   #[0:len(reference_sentences)]
        score1 = calucate_sbert(model1, query_sentences, use_refs)
        score2 = calucate_sbert(model2, query_sentences, use_refs)
        bleu_score = bleu.compute(predictions=query_sentences, references=use_refs.tolist())
        file.write(f"{obj['file']}, {obj['dataset']} ,{score1}, {score2}, {bleu_score['bleu']}" + '\n') 