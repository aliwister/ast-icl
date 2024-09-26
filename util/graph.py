
import torch
from torch_geometric.data import Data
import networkx as nx
from math import ceil

import ast
import re

import matplotlib.pyplot as plt



def plot_graph(graph, tokenizer):
    G = nx.DiGraph()
    x = tokenizer.batch_decode(graph.x, skip_special_tokens = True)
    edge_index = graph.edge_index.t()
    #print(x)
    # Add nodes and edges to the NetworkX graph
    for i in range(len(x)):
        #print(x[i])
        G.add_node(i, label=x[i])
    
    for edge in edge_index.numpy():
        print(edge)
        G.add_edge(edge[0], edge[1])
    
    # Get node labels
    node_labels = nx.get_node_attributes(G, 'label')
    
    # Draw the graph
    pos = nx.spring_layout(G)  # You can use other layout algorithms as well
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=500, font_size=8, arrowstyle='-|>', arrowsize=10)
    
    # Show the plot
    plt.show()

def parse_string(input_string):
    # Define the regular expression pattern
    pattern = r"^(\w+)\[['\"](.+?)['\"]\]$"
    # Match the pattern with the input string
    match = re.match(pattern, input_string)
    if match:
        # Extract the groups: command and content inside the brackets
        command = match.group(1)
        content = match.group(2)
        return command, content
    else:
        print('Shouldnt return none', input_string)
        return None, None

def parse_number_after_hash(input_string):
    # Define a regex pattern to match a # followed by one or more digits
    pattern = r"#(\d+)"

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # If a match is found, extract the number and convert it to an integer
    if match:
        number_str = match.group(1)  # group(1) returns the part of the string that matches the group (digits)
        number = int(number_str)
        return number
    else:
        # If no match is found, return None or handle the case as needed
        
        return None

def create_graph(input, tokenizer, idx):
    def tokenize(x):
        input = tokenizer(x, return_tensors="pt",  padding='max_length', max_length=50)
        input_ids = input.input_ids
        mask = input.attention_mask
        return input_ids, mask

    edge_index = []
    node_features = []
    attention_masks = []
    code = ast.literal_eval(input)
    try:
        for i, c in enumerate(code):
            parsed = parse_string(c)
            if parsed[0] is None:
                continue
            op = parsed[0]
            vals = parsed[1].split("', '")
            #print (op, vals)
            edge_index.append([i*2, i*2+1])

            input_ids, mask = tokenize([parsed[0]])
            node_features.append(input_ids)
            attention_masks.append(mask)
            #print(len(vals))
            if (len(vals) > 1):
                for j in vals:
                    #print(j)
                    if (j.startswith("#")):
                        k = parse_number_after_hash(j[0:2]) # Ignore text after the number for now
                        edge_index.append([i*2, k*2-1])
            if(len(vals[-1]) > 0):
                input_ids, mask = tokenize(vals[-1])
                node_features.append(input_ids)
                attention_masks.append(mask)
    except Exception as ex:
        print(ex)
        print("Error: ", input)
    #print(node_features)
    graph = Data(edge_index=torch.tensor(edge_index, dtype=torch.long).t(), x=torch.stack(node_features).squeeze(), attention_mask=torch.stack(attention_masks).squeeze(), idx=idx)
    return graph

    