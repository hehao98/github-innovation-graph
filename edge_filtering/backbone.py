'''
This module implements the disparity filter to compute a significance score of edge weights in networks
'''

import networkx as nx
import numpy as np
from scipy import integrate
import pandas as pd
import argparse

def disparity_filter(G, weight='weight'):
    ''' Compute significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009
        Args
            G: Weighted NetworkX graph
        Returns
            Weighted graph with a significance score (alpha) assigned to each edge
        References
            M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    '''
    
    if nx.is_directed(G): #directed case    
        N = nx.DiGraph()
        for u in G:
            
            k_out = G.out_degree(u)
            k_in = G.in_degree(u)
            
            if k_out > 1:
                sum_w_out = sum(np.absolute(G[u][v][weight]) for v in G.successors(u))
                for v in G.successors(u):
                    w = G[u][v][weight]
                    p_ij_out = float(np.absolute(w))/sum_w_out
                    alpha_ij_out = 1 - (k_out-1) * integrate.quad(lambda x: (1-x)**(k_out-2), 0, p_ij_out)[0]
                    N.add_edge(u, v, weight = w, alpha_out=float('%.4f' % alpha_ij_out))
                    
            # elif k_out == 1 and G.in_degree(G.successors(u)[0]) == 1:
            elif k_out == 1:
                successors = list(G.successors(u))
                if len(successors) == 1 and G.in_degree(successors[0]) == 1:
                    #we need to keep the connection as it is the only way to maintain the connectivity of the network
                    v = G.successors(u)[0]
                    w = G[u][v][weight]
                    N.add_edge(u, v, weight = w, alpha_out=0., alpha_in=0.)
                    #there is no need to do the same for the k_in, since the link is built already from the tail
            
            if k_in > 1:
                sum_w_in = sum(np.absolute(G[v][u][weight]) for v in G.predecessors(u))
                for v in G.predecessors(u):
                    w = G[v][u][weight]
                    p_ij_in = float(np.absolute(w))/sum_w_in
                    alpha_ij_in = 1 - (k_in-1) * integrate.quad(lambda x: (1-x)**(k_in-2), 0, p_ij_in)[0]
                    N.add_edge(v, u, weight = w, alpha_in=float('%.4f' % alpha_ij_in))
        return N
    
    else: #undirected case
        B = nx.Graph()
        for u in G:
            k = len(G[u])
            if k > 1:
                sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
                for v in G[u]:
                    w = G[u][v][weight]
                    p_ij = float(np.absolute(w))/sum_w
                    alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
                    B.add_edge(u, v, weight = w, alpha=float('%.4f' % alpha_ij))
        return B

def disparity_filter_alpha_cut(G,weight='weight',alpha_t=0.4, cut_mode='or'):
    ''' Performs a cut of the graph previously filtered through the disparity_filter function.
        
        Args
        ----
        G: Weighted NetworkX graph
        
        weight: string (default='weight')
            Key for edge data used as the edge weight w_ij.
            
        alpha_t: double (default='0.4')
            The threshold for the alpha parameter that is used to select the surviving edges.
            It has to be a number between 0 and 1.
            
        cut_mode: string (default='or')
            Possible strings: 'or', 'and'.
            It works only for directed graphs. It represents the logic operation to filter out edges
            that do not pass the threshold value, combining the alpha_in and alpha_out attributes
            resulting from the disparity_filter function.
            
            
        Returns
        -------
        B: Weighted NetworkX graph
            The resulting graph contains only edges that survived from the filtering with the alpha_t threshold
    
        References
        ---------
        .. M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    '''    
    
    
    if nx.is_directed(G):#Directed case:   
        B = nx.DiGraph()
        for u, v, w in G.edges(data=True):
            try:
                alpha_in =  w['alpha_in']
            except KeyError: #there is no alpha_in, so we assign 1. It will never pass the cut
                alpha_in = 1
            try:
                alpha_out =  w['alpha_out']
            except KeyError: #there is no alpha_out, so we assign 1. It will never pass the cut
                alpha_out = 1  
            
            if cut_mode == 'or':
                if alpha_in<alpha_t or alpha_out<alpha_t:
                    B.add_edge(u,v, weight=w[weight])
            elif cut_mode == 'and':
                if alpha_in<alpha_t and alpha_out<alpha_t:
                    B.add_edge(u,v, weight=w[weight])
        return B

    else:
        B = nx.Graph()#Undirected case:   
        for u, v, w in G.edges(data=True):
            
            try:
                alpha = w['alpha']
            except KeyError: #there is no alpha, so we assign 1. It will never pass the cut
                alpha = 1
                
            if alpha<alpha_t:
                B.add_edge(u,v, weight=w[weight])
        return B                
            
def find_optimal_alpha(G, target_percentage, max_iter=1000):
    """
    Determines the optimal alpha value for filtering a graph based on a target edge retention percentage.

    This function uses a proportional approach to dynamically adjust the alpha value. It decrements alpha 
    based on the difference between the current and target edge counts, aiming to find the alpha value that 
    results in a filtered graph with a number of edges closest to the target edge count specified by the threshold.

    Args:
    ----
    G: Weighted NetworkX graph
        The graph for which the optimal alpha value is to be determined.

    target_percentage: float
        The target percentage of edges to retain in the graph. It should be a value between 0 and 1.
        A target_percentage of 1.0 implies retaining all edges, while 0.1 would aim to retain 10% of the edges.

    max_iter: int (default=1000)
        The maximum number of iterations to run the proportional adjustment process. This limits the computation 
        time and ensures the function terminates.

    Returns:
    -------
    closest_alpha: float
        The calculated optimal alpha value that results in a graph with the number of edges closest to the 
        target edge count based on the specified target_percentage.

    """
    if target_percentage == 1.0:
        return 1  # Retain all edges

    total_edges = G.number_of_edges()
    target_edges = int(total_edges * target_percentage)
    alpha = 1

    closest_alpha = alpha
    closest_edge_count = total_edges
    closest_diff = abs(closest_edge_count - target_edges)

    for _ in range(max_iter):
        filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if ('alpha_in' in d and d['alpha_in'] < alpha) or ('alpha_out' in d and d['alpha_out'] < alpha)]
        current_edge_count = len(filtered_edges)
        current_diff = abs(current_edge_count - target_edges)

        if current_diff <= closest_diff:
            closest_diff = current_diff
            closest_edge_count = current_edge_count
            closest_alpha = alpha
        else:
            break  # Stop if we start moving away from the target

        # Proportional step size
        step_size = max(0.001, 0.1 * (current_diff / total_edges))
        alpha -= step_size

        if alpha < 0:
            alpha = 0  # Ensure alpha does not go below 0

    return closest_alpha

def load_data(file_path):
    return pd.read_csv(file_path)

def filter_data(data, year, quarters):
    return data[(data['year'] == year) & (data['quarter'].isin(quarters))]

def load_data(file_path):
    return pd.read_csv(file_path)

def filter_data(data, year, quarters):
    return data[(data['year'] == year) & (data['quarter'].isin(quarters))]

def normalize_weights(df, mode='outgoing'):
    """
    Normalizes the weights of edges in the DataFrame based on the total outgoing or incoming connections.

    Args:
    ----
    df: DataFrame
        The DataFrame containing the edges with their original weights.

    mode: str
        The mode of normalization. Can be 'outgoing', 'incoming', or 'none'.
        'outgoing' normalizes based on the sum of weights of outgoing edges for each source node.
        'incoming' normalizes based on the sum of weights of incoming edges for each destination node.
        'none' returns the DataFrame without any changes.

    Returns:
    -------
    DataFrame
        A modified DataFrame with normalized weights.
    """

    if mode == 'none':
        # No normalization needed, return original DataFrame
        return df

    # Make a copy to ensure we are not modifying the view directly 
    normalized_df = df.copy()

    if mode == 'outgoing':
        # Group by source and sum the weights
        total_weights = df.groupby('source')['weight'].sum().reset_index()
        total_weights.rename(columns={'weight': 'total_outgoing_weight'}, inplace=True)
        normalized_df = normalized_df.merge(total_weights, on='source')
        normalized_df['weight'] = normalized_df['weight'] / normalized_df['total_outgoing_weight']

    elif mode == 'incoming':
        # Group by destination and sum the weights
        total_weights = df.groupby('destination')['weight'].sum().reset_index()
        total_weights.rename(columns={'weight': 'total_incoming_weight'}, inplace=True)
        normalized_df = normalized_df.merge(total_weights, on='destination')
        normalized_df['weight'] = normalized_df['weight'] / normalized_df['total_incoming_weight']

    # Drop the total weights columns used for normalization
    normalized_df.drop(columns=['total_outgoing_weight', 'total_incoming_weight'], errors='ignore', inplace=True)

    return normalized_df

def process_entity_merging(df, merge_eu, merge_cn_hk, exclude_country=None):
    """
    Processes the DataFrame to handle entity merging for EU and HK-CN, including self-loop handling,
    and excluding a specified country if needed.

    Args:
    ----
    df: DataFrame
        The DataFrame containing the network data.
    merge_eu: bool
        Flag indicating whether to use the existing 'EU' entity.
    merge_cn_hk: bool
        Flag indicating whether to merge Hong Kong with China.
    exclude_country: str, optional
        The country code to exclude from the DataFrame.

    Returns:
    -------
    DataFrame
        The modified DataFrame with the specified entity merges, self-loop handling, and country exclusion.
    """
    # Make a copy to ensure we are not modifying the view directly 
    df = df.copy()

    if merge_eu:
        # List of EU country codes
        eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
        
        # Remove edges involving individual EU countries
        df = df[~df['source'].isin(eu_countries) & ~df['destination'].isin(eu_countries)]

    if merge_cn_hk:
        # Merge Hong Kong with China
        df.replace({'HK': 'CN'}, inplace=True)

    if exclude_country:
        # Exclude the specified country
        df = df[(df['source'] != exclude_country) & (df['destination'] != exclude_country)]

    # Handle self-loops after merging
    self_loops = df[df['source'] == df['destination']]
    if not self_loops.empty:
        # Aggregate weights of self-loops
        aggregated_self_loops = self_loops.groupby(['source', 'year', 'quarter']).agg({'weight': 'sum'}).reset_index()
        df = df[df['source'] != df['destination']]
        df = pd.concat([df, aggregated_self_loops], ignore_index=True)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Collaboration Network Filter')
    parser.add_argument('--inputFilePath', required=True, help='Location of input Collaboration Network Data edgelist')
    parser.add_argument('--outputFilePath', required=True, help='Location of output filtered edgelist')
    parser.add_argument('--year', type=int, choices=[2020, 2021, 2022, 2023], required=True, help='Year to filter data')
    parser.add_argument('--quarters', type=int, nargs='+', choices=[1, 2, 3, 4], required=True, help='Quarters to filter data')
    parser.add_argument('--threshold', type=float, required=True, help='Percentage of edges to retain (e.g., 0.10 for 10%)')
    parser.add_argument('--normalize', choices=['outgoing', 'incoming', 'none'], default='none', help='Normalize weights by outgoing or incoming totals')
    parser.add_argument('--mergeEU', action='store_true', help='Merge all EU countries into a single node')
    parser.add_argument('--mergeCNHK', action='store_true', help='Combine Hong Kong with China')
    parser.add_argument('--excludeUS', action='store_true', help='Exclude the US from the network')

    args = parser.parse_args()

    # Load and filter data
    input_file_path = args.inputFilePath
    df = load_data(input_file_path)
    filtered_df = filter_data(df, args.year, args.quarters)

    exclude_country = 'US' if args.excludeUS else None
    filtered_df = process_entity_merging(filtered_df, args.mergeEU, args.mergeCNHK, exclude_country)

    # Normalize weights if required
    if args.normalize != 'none':
        filtered_df = normalize_weights(filtered_df, mode=args.normalize)

    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(filtered_df, source='source', target='destination', edge_attr='weight', create_using=Graphtype)
    G = disparity_filter(G)

    target_percentage = args.threshold  # Use the provided threshold argument
    optimal_alpha = find_optimal_alpha(G, target_percentage)

    # Filter the graph using the optimal alpha
    if nx.is_directed(G):
        G2 = nx.DiGraph([(u, v, d) for u, v, d in G.edges(data=True) if ('alpha_in' in d and d['alpha_in'] < optimal_alpha) or ('alpha_out' in d and d['alpha_out'] < optimal_alpha)])
    else:
        G2 = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if 'alpha' in d and d['alpha'] < optimal_alpha])

    print('optimal alpha =', optimal_alpha)
    print('original: nodes = %s, edges = %s' % (G.number_of_nodes(), G.number_of_edges()))
    print('backbone: nodes = %s, edges = %s' % (G2.number_of_nodes(), G2.number_of_edges()))

    # Convert G2 back to DataFrame
    edges_df = nx.to_pandas_edgelist(G2)

    # Merge with the original dataframe using inner join to keep original columns
    merged_df = pd.merge(filtered_df, edges_df,  how='inner', left_on=['source','destination'], right_on = ['source','target'])
    merged_df.drop(columns=['target','weight_y'], inplace=True)
    merged_df = merged_df.rename(columns={"weight_x": "weight"})

    # Write DataFrame to CSV
    output_file_path = args.outputFilePath
    merged_df.to_csv(output_file_path, index=False)

    print(f"Filtered graph data written to {output_file_path}")

if __name__ == '__main__':
    main()