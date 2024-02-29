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
                    v = successors[0] 
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

def load_data(file_path):
    return pd.read_csv(file_path)

def aggregate_edges(data):
    """
    Aggregates the weights of edges across different years and quarters.
    Sums up the weights for the same edge across all years and quarters.
    """
    # Group by source and destination and sum the weights
    aggregated_data = data.groupby(['source', 'destination']).agg({'weight': 'sum'}).reset_index()
    return aggregated_data

def filter_data(data, year=None, quarters=None):
    """
    Filter data based on the provided year and quarters.
    If year or quarters are None, include and aggregate all data.
    """
    if year is not None and quarters is not None:
        filtered_data = data[(data['year'] == year) & (data['quarter'].isin(quarters))]
    else:
        filtered_data = data  # Include all data if year or quarters are not specified

    # Aggregate weights for the same edge across different years and quarters
    return aggregate_edges(filtered_data)

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
        'log' transforms the weights using the log function.
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

    elif mode == 'log':
        normalized_df['weight'] = np.log(normalized_df['weight'])

    # Drop the total weights columns used for normalization
    normalized_df.drop(columns=['total_outgoing_weight', 'total_incoming_weight'], errors='ignore', inplace=True)

    return normalized_df

def process_entity_merging(df, merge_eu, merge_cn_hk, exclude_countries=None):
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

    # Exclude specified countries
    if exclude_countries:
        for country in exclude_countries:
            df = df[(df['source'] != country) & (df['destination'] != country)]

    # Handle self-loops after merging
    self_loops = df[df['source'] == df['destination']]
    if not self_loops.empty:
        # Aggregate weights of self-loops
        aggregated_self_loops = self_loops.groupby(['source', 'year', 'quarter']).agg({'weight': 'sum'}).reset_index()
        df = df[df['source'] != df['destination']]
        df = pd.concat([df, aggregated_self_loops], ignore_index=True)
    
    return df

def create_output_file_name(base_path, alpha):
    '''
    Create a file name by appending the alpha value to the base file path.

    Args:
        base_path (str): Base file path.
        alpha (float): Alpha value.

    Returns:
        str: Constructed file name with alpha value appended.
    '''
    alpha_str = str(alpha).replace('.', '_')
    return f"{base_path}_alpha_{alpha_str}.csv"

def filter_graph_by_alpha(G, alpha):
    '''
    Filters the given graph based on the optimal alpha value.

    Args:
        G (NetworkX graph): Input graph.
        alpha (float): Optimal alpha value for filtering.

    Returns:
        NetworkX graph: Filtered graph based on the alpha value.
    '''
    if nx.is_directed(G):
        G_filtered = nx.DiGraph([(u, v, d) for u, v, d in G.edges(data=True) if ('alpha_in' in d and d['alpha_in'] < alpha) or ('alpha_out' in d and d['alpha_out'] < alpha)])
    else:
        G_filtered = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if 'alpha' in d and d['alpha'] < alpha])

    return G_filtered

def print_graph_stats(G_original, G_filtered, alpha):
    '''
    Print statistics of the original and filtered graph.

    Args:
        G_original (NetworkX graph): Original graph.
        G_filtered (NetworkX graph): Filtered graph.
        alpha (float): Alpha value used for filtering.
    '''
    print(f'Optimal alpha = {alpha}')
    print(f'Original: nodes = {G_original.number_of_nodes()}, edges = {G_original.number_of_edges()}')
    print(f'Filtered: nodes = {G_filtered.number_of_nodes()}, edges = {G_filtered.number_of_edges()}')

def save_filtered_graph(G_filtered, df_original, file_name):
    '''
    Save the filtered graph to a CSV file.

    Args:
        G_filtered (NetworkX graph): Filtered graph.
        df_original (DataFrame): Original DataFrame.
        file_name (str): File name for saving the CSV.
    '''
    edges_df = nx.to_pandas_edgelist(G_filtered)
    merged_df = pd.merge(df_original, edges_df, how='inner', left_on=['source', 'destination'], right_on=['source', 'target'])
    merged_df.drop(columns=['target', 'weight_y'], inplace=True)
    merged_df = merged_df.rename(columns={"weight_x": "weight"})
    merged_df = merged_df[['source', 'destination', 'weight', 'alpha_out', 'alpha_in']]
    merged_df.to_csv(file_name, index=False)
    print(f"Filtered graph data written to {file_name}")

def main():
    parser = argparse.ArgumentParser(description='Collaboration Network Filter')
    parser.add_argument('--inputFilePath', required=True, help='Location of input Collaboration Network Data edgelist')
    parser.add_argument('--outputFilePath', required=True, help='Location of output filtered edgelist')
    parser.add_argument('--normalize', choices=['outgoing', 'incoming', 'log', 'none'], default='none', help='Normalize weights by outgoing or incoming totals')
    parser.add_argument('--mergeEU', action='store_true', help='Merge all EU countries into a single node')
    parser.add_argument('--mergeCNHK', action='store_true', help='Combine Hong Kong with China')
    parser.add_argument('--excludeCountries', nargs='*', type=str, default=[], help='List of country codes to exclude from the network')
    parser.add_argument('--optimalAlpha', nargs='*', type=float, default=[0.09], help='List of optimal alpha values')
    args = parser.parse_args()

    # Load and filter data
    df = load_data(args.inputFilePath)
    filtered_df = filter_data(df)
    filtered_df = process_entity_merging(filtered_df, args.mergeEU, args.mergeCNHK, exclude_countries=args.excludeCountries)

    # Normalize weights if required
    if args.normalize != 'none':
        filtered_df = normalize_weights(filtered_df, mode=args.normalize)

    # Create graph from edge list
    G = nx.from_pandas_edgelist(filtered_df, source='source', target='destination', edge_attr='weight', create_using=nx.DiGraph())
    G = disparity_filter(G)

    for optimal_alpha in args.optimalAlpha:
        output_file_name = create_output_file_name(args.outputFilePath, optimal_alpha)

        # Filter the graph based on the optimal alpha
        G_filtered = filter_graph_by_alpha(G, optimal_alpha)

        print_graph_stats(G, G_filtered, optimal_alpha)

        # Convert filtered graph to DataFrame and save
        save_filtered_graph(G_filtered, filtered_df, output_file_name)

if __name__ == '__main__':
    main()
