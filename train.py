import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from argparse import ArgumentParser
import pandas as pd
from torch_geometric.loader import DataLoader
#from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from math import ceil
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI


from model.gnn_encoder import GNNModel
import evaluate
from run_prompts import run_prompts
from util.ast_icl import GCN, create_vocab, create_graph
from util.dataset import load_cosql_dataset, load_dataset, load_spider_dataset
from util.prompt import create_cot_prompt, create_incontext_prompt2, create_justcode_prompt, create_zeroshot_prompt, get_answer
from train import encode
import pdb

from torch_geometric.data import Data #, Daparse_querytaLoader
from util.sql_tree import parse_query
from rank_bm25 import BM25Okapi
from sklearn.cluster import AgglomerativeClustering
from nltk.tokenize import word_tokenize
import string
import time



device = 'cuda' if torch.cuda.is_available() else 'cpu'
def process_string_to_array(string):
    stripped_string = string.strip('[]')
    array = np.array([float(num) for num in stripped_string.split()])
    return array


def cluster_data(df, n_clusters):
    kmeans = KMeans(n_clusters=45)
    labels_train = kmeans.fit_predict(list(df['data'].values))
    return kmeans, labels_train

def encode(self, batch, lang_model):
    x, edge_index, attention_mask = batch.x, batch.edge_index, batch.attention_mask
    #pdb.set_trace()
    inputs = lang_model(x, output_hidden_states=True, attention_mask=attention_mask)
    hidden_states = inputs.hidden_states
    last_hidden_state = hidden_states[-1]
    denom = torch.sum(attention_mask, -1, keepdim=True)
    feat = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / denom
    feat = feat.to(torch.float32)
    return feat

def get_samples(df, cluster, num):
    fdf = df
    if cluster > -1:
        fdf = df[df['label'] == cluster]
    sampled_data = fdf.sample(n=num)
    return list(chain.from_iterable(zip(sampled_data['query'], sampled_data['utterance'])))

def get_samples_top(df, cluster, num, train_pool, test):
    differences = cosine_similarity(train_pool, [test])
    x = np.argpartition(np.squeeze(differences), -num)[-num:]
    sampled_data = df.iloc[x]
    return list(chain.from_iterable(zip(sampled_data['query'], sampled_data['utterance'])))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    return tokens


def get_samples_bm25(df, cluster, num, bm25, test):
    tokenized_query = preprocess(test)

    doc_scores = bm25.get_scores(tokenized_query)
    x = np.argpartition(doc_scores, -num)[-num:]
    sampled_data = df.iloc[x]
    return list(chain.from_iterable(zip(sampled_data['query'], sampled_data['utterance'])))

def train(model, loader):
    all_pooled = []

    # Encode the graph and pool to a vector
    with torch.no_grad():  # We're not training, so no gradients needed
        for batch in loader:
            #data = Data(x=x, edge_index=edge_index)
            embeddings = model(batch)
            pooled = global_mean_pool(embeddings, batch.batch)  # Pool embeddings to a single vector
            all_pooled.append(pooled)
    all_pooled_tensor = torch.cat(all_pooled, dim=0)
    n_clusters = 20  # Adjust the number of clusters as needed
    all_pooled_numpy = all_pooled_tensor.cpu().numpy()
    return all_pooled_numpy

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cosql") 
    parser.add_argument('--num_examples', type=int, default=2)
    parser.add_argument('--method', type=str, default="icl-top")
    parser.add_argument('--output_csv', type=str, default='/home/ali.lawati/gnn-incontext/cosql_processed2.csv.gptj.incontext')
    parser.add_argument('--lang_model', type=str, default="EleutherAI/gpt-j-6B") 
    parser.add_argument('--limit', type=bool, default="False") 
    
    args = parser.parse_args()

    df_train, df_test = load_dataset(args.ataset)

    df_train[['features','edge_index']] = df_train['query'].apply(parse_query).apply(pd.Series)
    df_test[['features','edge_index']] = df_test['query'].apply(parse_query).apply(pd.Series)

    # Embed and cluster training dataset:
    word_to_index, embeddings = create_vocab(df_train)
    #tokenized_nodes = [torch.tensor([word_to_index[word] for word in node]) for node in df_train['features']]

    n_clusters = 20 
    input_dim = 100  # Dimension of node features
    hidden_dim = 24  # Number of hidden units
    output_dim = 2  # Dimension of output embeddings
    train_graphs = df_train.apply(create_graph, axis=1, args=(word_to_index, embeddings))
    test_graphs = df_test.apply(create_graph, axis=1, args=(word_to_index, embeddings))

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    time_gnn = 0
    if (args.method == "BM25"):
        tokenized_corpus = [preprocess(text) for text in df_train['query']]
        bm25 = BM25Okapi(tokenized_corpus)
    else:    
        start_time = time.time()
        model = GCN(input_dim, hidden_dim, output_dim)
        train_pool = train(model, train_loader)
        test_pool = train(model, test_loader)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_pool)
        df_train['label'] = kmeans.labels_
        test_pool_clusters = kmeans.predict(test_pool)
        end_time = time.time()
        time_gnn = end_time - start_time
    predictions1, predictions2, predictions3, references = [], [], [], []
    prompts1, prompts2, prompts3, prompts4 = [], [], [], []

    batch_size = 16
    start_time = time.time()

    for i in range(0, len(df_test)):
        prog = df_test.iloc[i]['query']
        if(args.method == "random"):
            p_args1 = get_samples(df_train, -1, args.num_examples) + [prog]
            prompt1 = create_incontext_prompt2(*p_args1)
        elif(args.method=="BM25"):
            p_args1 = get_samples_bm25(df_train, -1, args.num_examples, bm25, prog) + [prog]
            prompt1 = create_incontext_prompt2(*p_args1)
        elif(args.method=="icl-top"):
            cluster = test_pool_clusters[i]
            p_args1 = get_samples_top(df_train, cluster, args.num_examples, train_pool, test_pool[i]) + [prog]
            prompt1 = create_incontext_prompt2(*p_args1)
        elif(args.method=="zero"):
            p_args2 = (prog,)
            prompt1 = create_zeroshot_prompt(*p_args2)
        elif(args.method=="cot"):
            p_args2 = (prog,)
            prompt1 = create_cot_prompt(*p_args2)
        elif(args.method=="zero-gpt"):
            prompt1 = prog
        else:
            cluster = test_pool_clusters[i]
            p_args1 = get_samples(df_train, cluster, args.num_examples) + [prog]
            prompt1 = create_incontext_prompt2(*p_args1)

        prompts1.append(prompt1)
        references.append(df_test.iloc[i]['utterance'])

    data_dict = {
        'prompt1': np.squeeze(prompts1),      # First array as the 'ID' column
        'ref': references      # Third array as the 'Age' column
    }
    end_time = time.time()
    time_prompt = end_time - start_time



    df = pd.DataFrame(data_dict)

    #if args.limit:
    #    df = df[0:10]
    #    references = references[0:10]
    

    prompts_all = df['prompt1'].to_numpy().flatten()
    if (args.lang_model == "openai/chatgpt"):
        client = OpenAI(
        )   
        def get_response_zero(prompt):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an intelligent SQL Code assistant who effectively translates the intent and logic of the SQL queries into natural language that is easy to understand."},
                    {"role": "user", "content": """Convert the given SQL query into a clear and concise natural language query limited to 1 sentence.  Ensure that the request accurately represents the actions specified in the SQL query and is easy to understand for someone without technical knowledge of SQL.
                     Input: """ + prompt},
                ],
                temperature=1,
                max_tokens=150,
                top_p=1
            )
            return response.choices[0].message.content

        def get_response(prompt):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                max_tokens=150,
                top_p=1
            )
            return response.choices[0].message.content


        responses = []
        for prompt in prompts_all:
            if (args.method == "zero-gpt"):
                response = get_response_zero(prompt)
            else:
                response = get_response(prompt)
            print(f"Prompt: {prompt}\nResponse: {response}\n")
            responses.append(response)
        df = pd.DataFrame(responses)
        df.to_csv(f"./output/{args.dataset}-{args.lang_model.split('/')[1]}-{args.method}-{args.num_examples}.csv", index=False)
#    else:
#        run_prompts(args.lang_model, prompts_all, f"./output/{args.dataset}-{args.lang_model.split('/')[1]}-{args.method}-{args.num_examples}.csv") #args.output_csv)
    df.to_csv(f"./prompt-files/{args.dataset}-{args.method}-{args.num_examples}.csv", index=False)
