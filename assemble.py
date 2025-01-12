import numpy as np
import torch
from itertools import chain
from argparse import ArgumentParser
import pandas as pd
from torch_geometric.loader import DataLoader
#from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import global_mean_pool

from util.ast_icl_sage import SAGE
from util.ast_icl_gcn import GCN
from util.ast_icl_gat import GAT

from util.dataset import load_orig_dataset, load_new_test_dataset
from util.prompt import create_cot_prompt, create_incontext_prompt2, create_zeroshot_prompt
import os

from torch_geometric.data import Data #, Daparse_querytaLoader
from util.sql_tree import parse_query
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
import time



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_vocab(df):
    word_embedding_dim = 100  # Dimension of word embeddings
    word_to_index = {}  # Dictionary to map words to indices

    embeddings = torch.nn.Embedding(len(df['features'].explode().unique()), word_embedding_dim) 
    for i, word in enumerate(df['features'].explode().unique()):
        word_to_index[word] = i
    
    return word_to_index, embeddings

def create_graph(row, word_to_index, embeddings):

    token_features = [word_to_index[word] for word in row['features']]
    embed_features = embeddings(torch.tensor(token_features))
    graph = Data(edge_index=torch.tensor(row['edge_index'], dtype=torch.long).t(), x=embed_features, idx=row.name)
    return graph


def create_input_file(prompts, refs, num_examples, method, task_name):
    data_dict = {
        'prompt': prompts
    }
    df = pd.DataFrame(data_dict)
    output_dir = f"input/{task_name}/{method}"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{method}-{num_examples}.csv", index=False)


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
    sampled_data = fdf.sample(n=num, replace=True)
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

def assemble(method, dataset, model_name):
    df_train, _ = load_orig_dataset(dataset)
    df_test = load_new_test_dataset(dataset)

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
    if (method == "BM25"):
        tokenized_corpus = [preprocess(text) for text in df_train['query']]
        bm25 = BM25Okapi(tokenized_corpus)
    else:    
        start_time = time.time()
        if (model_name == "GCN"):
            model = GCN(input_dim, hidden_dim, output_dim)
        elif (model_name == "GAT"):
            model = GAT(input_dim, hidden_dim, output_dim)
        elif (model_name == "SAGE"):
            model = SAGE(input_dim, hidden_dim, output_dim)
        else:
            raise ValueError("Choose a model")

        train_pool = train(model, train_loader)
        test_pool = train(model, test_loader)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_pool)
        df_train['label'] = kmeans.labels_
        test_pool_clusters = kmeans.predict(test_pool)
        end_time = time.time()
        time_gnn = end_time - start_time


    LEN = 10
    df = df_test
    refs = []
    prompts = [[] for i in range(0,LEN)]
    indices = [[] for i in range(0,LEN)]
    batch_size = 16
    start_time = time.time()

    for i in range(0, len(df_test)):
        prog = df_test.iloc[i]['query']
        for n in range(0, LEN):
            if(method == "random"):
                samples = get_samples(df_train, -1, n+1) + [prog]
                prompt1 = create_incontext_prompt2(*samples)
            elif(method=="BM25"):
                samples = get_samples_bm25(df_train, -1, n+1, bm25, prog) + [prog]
                prompt1 = create_incontext_prompt2(*samples)
            elif(method=="icl-top"):
                cluster = test_pool_clusters[i]
                samples = get_samples_top(df_train, cluster, n+1, train_pool, test_pool[i]) + [prog]
                prompt1 = create_incontext_prompt2(*samples)
            elif(method=="zero"):
                p_args2 = (prog,)
                prompt1 = create_zeroshot_prompt(*p_args2)
            elif(method=="cot"):
                p_args2 = (prog,)
                prompt1 = create_cot_prompt(*p_args2)
            elif(method=="zero"):
                prompt1 = prog
            else:
                cluster = test_pool_clusters[i]
                samples = get_samples(df_train, cluster, n+1) + [prog]
                prompt1 = create_incontext_prompt2(*samples)

            prompts[n].append(prompt1)
        #refs.append(df_test.iloc[i]['utterances'])

    [create_input_file(prompts[i], refs, i+1, f"{method}-{dataset}-exp", dataset+f"-{model_name}") for i in [1,3,7]]



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cosql") 
    parser.add_argument('--num_examples', type=int, default=2)
    parser.add_argument('--method', type=str, default="icl-top")
    parser.add_argument('--limit', type=bool, default="False") 
    parser.add_argument('--model', type=str, default="SAGE") 
    
    args = parser.parse_args()
    assemble(args.method, args.dataset, args.model)
    #datasets = ['spider', 'sparc', 'cosql']
    #methods = ['icl-top']#,'icl']
    #models = ['SAGE']#, 'GAT']
    #for d in datasets:
    #    for m in methods:
    #        for m2 in models:
    #            assemble(m, d, m2)
