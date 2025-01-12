from sqlglot import parse_one


def parse_query(query):
    ast = parse_one(query)
    idx = 0
    features = []
    edge_index = []
    mapping = {}
    for node in ast.walk():
        features.append(type(node).__name__)
        mapping[id(node)] = idx
        if (node.depth > 0):
            parent = mapping[id(node.parent)]
            edge_index.append([parent, idx])
        idx = idx + 1
    return features, edge_index