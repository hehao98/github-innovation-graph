import pandas as pd

# Read the CSV file
input_file = '/Users/katy/github-innovation-graph/data/economy_collaborators.csv'
df = pd.read_csv(input_file)

# Ensure 'source' and 'destination' are strings
df['source'] = df['source'].astype(str)
df['destination'] = df['destination'].astype(str)

# Create a dictionary to store the minimum weights for undirected edges
undirected_edges = {}

# Iterate over the rows in the DataFrame
for _, row in df.iterrows():
    # Sort the nodes to treat the edge as undirected
    nodes = tuple(sorted([row['source'], row['destination']]))
    
    # Update the weight if it's smaller than the existing one, or if the edge is new
    if nodes not in undirected_edges or row['weight'] < undirected_edges[nodes]:
        undirected_edges[nodes] = row['weight']

# Create a new DataFrame for the undirected edges
undirected_df = pd.DataFrame(
    [(src, dest, weight) for (src, dest), weight in undirected_edges.items()],
    columns=['source', 'target', 'weight']
)

# Save the undirected edges to a new CSV file
output_file = '/Users/katy/github-innovation-graph/data/undirected_economy_collaborators.csv'
undirected_df.to_csv(output_file, index=False)

print(f'Undirected edges saved to {output_file}')
