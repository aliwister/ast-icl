import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_new_test_dataset(dataset):
    mult_ref_json = f"{Path.home()}/ast-datasets/{dataset}-ast.test.json"
    with open(mult_ref_json, 'r') as file:
        data = json.load(file)
    refs = [x['utterances'] for x in data]
    queries = [x['query'] for x in data]
    return pd.DataFrame({'utterances': refs, 'query': queries})

def load_new_test_w_querytype(dataset):
    mult_ref_json = f"{Path.home()}/ast-datasets/with_query_type/{dataset}.w_type.new.test.json"
    with open(mult_ref_json, 'r') as file:
        data = json.load(file)
    refs = [x['utterances'] for x in data]
    queries = [x['query'] for x in data]
    types = [x['query_type'] for x in data]
    return pd.DataFrame({'utterances': refs, 'query': queries, 'query_type': types})


def load_orig_dataset(dataset):
    if (dataset == "cosql"):
        df_train, df_test = load_cosql_dataset()
    elif(dataset == "spider"):
        df_train, df_test = load_spider_dataset()
    elif (dataset == "sparc"):
        df_train, df_test = load_sparc_dataset()
    elif (dataset == 'kaggledbqa'):
        df_train, df_test = load_kaggledbqa_dataset()
    return df_train, df_test

def load_kaggledbqa_dataset():
    dataset_train = f'{Path.home()}/text2sql-datasets/KaggleDBQA/kaggle_dbqa.train.json'
    dataset_test = f'{Path.home()}/text2sql-datasets/KaggleDBQA/kaggle_dbqa.test.json'
 
    # Open and read the JSON file
    with open(dataset_train, 'r') as file1:
        train = json.load(file1)
    with open(dataset_test, 'r') as file2:
        test = json.load(file2)
    # Print the contents of the JSON file
    df_train = pd.DataFrame(train, columns=['question', 'query'])
    df_test = pd.DataFrame(test, columns=['question', 'query'])
    df_train['utterance'] = df_train['question']
    df_test['utterance'] = df_test['question']

    #df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)

    return df_train, df_test

def load_cosql_dataset():
    dataset_train = f'{Path.home()}/text2sql-datasets/cosql/sql_state_tracking/cosql_train.json'
    dataset_test = f'{Path.home()}/text2sql-datasets/cosql/sql_state_tracking/cosql_dev.json'
 
    # Open and read the JSON file
    with open(dataset_train, 'r') as file1:
        train = json.load(file1)
    with open(dataset_test, 'r') as file2:
        test = json.load(file2)
    # Print the contents of the JSON file
    df_train = pd.DataFrame([item["final"] for item in train], columns=['utterance', 'query'])
    df_test = pd.DataFrame([item["final"] for item in test], columns=['utterance', 'query'])

    #df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)

    return df_train, df_test

def load_sparc_dataset():
    dataset_train = f'{Path.home()}/text2sql-datasets/sparc/train.json'
    dataset_test = f'{Path.home()}/text2sql-datasets/sparc/dev.json'
    # Open and read the JSON file
    with open(dataset_train, 'r') as file1:
        train = json.load(file1)
    with open(dataset_test, 'r') as file2:
        test = json.load(file2)
    # Print the contents of the JSON file
    df_train = pd.DataFrame([item["final"] for item in train], columns=['utterance', 'query'])
    df_test = pd.DataFrame([item["final"] for item in test], columns=['utterance', 'query'])

    return df_train, df_test
    

def load_spider_dataset():
    # Open and read the JSON file
    dataset_train = f"{Path.home()}/text2sql-datasets/spider/train_spider.json"
    dataset_test = f"{Path.home()}/text2sql-datasets/spider/dev.json"

    with open(dataset_train, 'r') as file1:
        train = json.load(file1)
    with open(dataset_test, 'r') as file2:
        test = json.load(file2)

    # Print the contents of the JSON file
    df_train = pd.DataFrame(train, columns=['question', 'query'])
    df_test = pd.DataFrame(test, columns=['question', 'query'])
    df_train['utterance'] = df_train['question']
    df_test['utterance'] = df_test['question']

    return df_train, df_test
