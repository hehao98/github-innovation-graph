import pandas as pd
import networkx as nx
from collections import OrderedDict


# Author: Hongbo Fang


def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference

    # Reference: https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy

    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g



# G = nx.DiGraph()
# G.add_edge(1, 2, weight = 3)
# G.add_edge(1, 3, weight = 1)
# G.add_edge(3, 2, weight = 3)

# print(G.in_degree([1,3, 2], weight = 'weight'))

# print(G.out_degree([1,3, 2], weight = 'weight'))
# print(nx.clustering(G, weight = 'weight'))


G = nx.DiGraph()
data = pd.read_csv()

for row_index, row in data.iterrows():
    source = data['source']
    target = data['target']

    weight = data['weight']

    G.add_edge(source, target, weight = weight)

list_of_nodes = list(G.nodes)

node2indegree = {node:degree for node, degree in G.in_degree(list_of_nodes)}

node2outdegree = {node:degree for node, degree in G.out_degree(list_of_nodes)}

node2cc = nx.clustering(G, weight = 'weight')


df_res = []
for node in node2cc:
    orderdict = OrderedDict()

    orderdict['node'] = node
    orderdict['in_degree']= node2indegree[node]
    orderdict['out_degree']= node2outdegree[node]
    orderdict['clustering_coefficient']= node2cc[node]
    df_res.append(orderdict)

df_res = pd.DataFrame(df_res)
df_res.to_csv()
print("size of df_res", len(df_res))
print("finished")