import os
import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import collections

# Include the functions: , load_data, filter_data, normalize_weights, process_entity_merging
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
    """
    Loads data from a given file path into a Pandas DataFrame.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def filter_data(data, year, quarters):
    """
    Filters the DataFrame based on a specific year and quarters.

    Args:
        data (pd.DataFrame): The DataFrame containing the network data. 
                             It should have 'year' and 'quarter' columns.
        year (int): The year to filter the data on. This should be a column in 'data'.
        quarters (list): A list of integers representing the quarters to filter the data on.
                         These should be values in the 'quarter' column of 'data'.

    Returns:
        pd.DataFrame: A DataFrame filtered based on the specified year and quarters.
    """
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

def largest_component_size(G):
    """ Returns the size of the largest connected component in the graph G. """
    if nx.is_directed(G):
        # largest_comp = max(nx.strongly_connected_components(G), key=len)
        largest_comp = max(nx.weakly_connected_components(G), key=len)

    else:
        largest_comp = max(nx.connected_components(G), key=len)
    return len(largest_comp)

def plot_cumulative_degree_distribution(G, alpha, ax, include_label):
    """
    Plots the cumulative degree distribution of a graph.

    Args:
        G (nx.Graph): NetworkX graph.
        alpha (float): Alpha value for the plot.
        ax (matplotlib.axes.Axes): Matplotlib Axes object.
        include_label (bool): Flag to include label in the plot.
    """
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cum_cnt = np.cumsum(cnt) / sum(cnt)
    label = f'α = {alpha}' if include_label else None
    ax.loglog(deg, cum_cnt, marker='o', linestyle='None', label=label)

def plot_link_weight_distribution(G, alpha, ax, include_label):
    """
    Plots the link weight distribution of a graph.

    Args:
        G (nx.Graph): NetworkX graph.
        alpha (float): Alpha value for the plot.
        ax (matplotlib.axes.Axes): Matplotlib Axes object.
        include_label (bool): Flag to include label in the plot.
    """
    weights = [d['weight'] for u, v, d in G.edges(data=True)]
    weights.sort(reverse=True)
    weight_counts = collections.Counter(weights)
    weight, freq = zip(*weight_counts.items())
    weight_freq_cum = np.cumsum(freq) / sum(freq)
    label = f'α = {alpha}' if include_label else None
    ax.loglog(weight, weight_freq_cum, marker='o', linestyle='None', label=label)

def plot_clustering_coefficient(G, alpha, ax, include_label, total_edges, total_weight, total_nodes, alpha_values, highlight_alpha=None):
    """
    Plots the clustering coefficient of a graph.

    Args:
        G (nx.Graph): NetworkX graph.
        alpha (float): Alpha value for the plot.
        ax (matplotlib.axes.Axes): Matplotlib Axes object.
        include_label (bool): Flag to include label in the plot.
        total_edges (int): Total number of edges in the original graph.
        total_weight (float): Total weight of the original graph.
        total_nodes (int): Total number of nodes in the original graph.
        alpha_values (list): List of alpha values.
        highlight_alpha (float, optional): Alpha value to be highlighted.
    """
    current_edges = G.number_of_edges()
    current_weight = sum(nx.get_edge_attributes(G, 'weight').values())
    current_nodes = G.number_of_nodes()

    percent_weight = current_weight / total_weight
    percent_nodes = current_nodes / total_nodes
    avg_clustering = sum(nx.clustering(G).values()) / current_nodes

    label_clustering = 'Clustering Coefficient' if include_label and alpha == min(alpha_values) else None
    label_weight = '% of Total Weight' if include_label and alpha == min(alpha_values) else None
    label_nodes = '% of Total Nodes' if include_label and alpha == min(alpha_values) else None

    ax.plot(alpha, avg_clustering, 'o-', color='red', label=label_clustering)
    ax.plot(alpha, percent_weight, 's-', color='green', label=label_weight)
    ax.plot(alpha, percent_nodes, 'd-', color='blue', label=label_nodes)

    # Highlight a specific alpha value if provided
    if highlight_alpha is not None:
        ax.axvline(x=highlight_alpha, color='purple', linestyle='--', label=f'α = {highlight_alpha}' if alpha == highlight_alpha else "")


def plot_largest_component_ratio(G, alpha, ax, original_lcc_size, include_label, highlight_alpha=None):
    """
    Plots the largest component ratio of a graph.

    Args:
        G (nx.Graph): NetworkX graph.
        alpha (float): Alpha value for the plot.
        ax (matplotlib.axes.Axes): Matplotlib Axes object.
        original_lcc_size (int): Size of the largest component in the original graph.
        include_label (bool): Flag to include label in the plot.
        highlight_alpha (float, optional): Alpha value to be highlighted.
    """
    lcc_size = largest_component_size(G)
    ratio = lcc_size / original_lcc_size
    label = f'α = {alpha}' if include_label else None
    ax.plot(alpha, ratio, 'x-', label=label)
    # Highlight a specific alpha value if provided
    if highlight_alpha is not None:
        ax.axvline(x=highlight_alpha, color='purple', linestyle='--', label=f'α = {highlight_alpha}' if alpha == highlight_alpha else "")

def analyze_network(G, alpha_values, year, quarters):
    """
    Analyzes the network for given alpha values and plots relevant graphs.

    Args:
        G (nx.Graph): NetworkX graph.
        alpha_values (list): List of alpha values for analysis.
        year (int): Year for analysis.
        quarters (list): List of quarters for analysis.
    """
    # Calculate the total weights and total nodes in the original graph
    total_weight = sum(nx.get_edge_attributes(G, 'weight').values())
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    sample_size = 5  # Choose the number of sample points for the legend
    sampled_alpha_values = np.linspace(min(alpha_values), max(alpha_values), sample_size)

    # Calculate the largest connected component size of the original graph
    original_lcc_size = largest_component_size(G)
    previous_lcc_ratio = 1
    largest_drop = 0
    alpha_before_largest_drop = None

    # Find the alpha before the largest drop
    for alpha in alpha_values:
        G_filtered = disparity_filter_alpha_cut(G, alpha_t=alpha)
        lcc_size = largest_component_size(G_filtered)
        lcc_ratio = lcc_size / original_lcc_size

        # Check for the largest drop in LCC size
        drop = previous_lcc_ratio - lcc_ratio
        if drop > largest_drop:
            largest_drop = drop
            alpha_before_largest_drop = alpha_values[alpha_values.index(alpha) - 1]

        previous_lcc_ratio = lcc_ratio

    fig, axs = plt.subplots(2, 2, figsize=(12, 12)) 
    for alpha in alpha_values:
        G_filtered = disparity_filter_alpha_cut(G, alpha_t=alpha)
        # Determine if this alpha value is in the sampled set for the legend
        include_label = alpha in sampled_alpha_values
        plot_cumulative_degree_distribution(G_filtered, alpha, axs[0, 0], include_label)
        plot_link_weight_distribution(G_filtered, alpha, axs[0, 1], include_label)
        plot_clustering_coefficient(G_filtered, alpha, axs[1, 0], include_label, total_edges, total_weight, total_nodes, alpha_values, highlight_alpha=alpha_before_largest_drop)
        plot_largest_component_ratio(G_filtered, alpha, axs[1, 1], original_lcc_size, include_label, highlight_alpha=alpha_before_largest_drop)

    for ax_row in axs:
        for ax in ax_row:
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')

    axs[0, 0].set_title('Cumulative Degree Distribution')
    axs[0, 0].set_xlabel('Degree')
    axs[0, 0].set_ylabel('Pc(k)')

    axs[0, 1].set_title('Distribution of Link Weights')
    axs[0, 1].set_xlabel('Weight')
    axs[0, 1].set_ylabel('P(ω)')

    axs[1, 0].set_title('Before and After Disparity Filter Ratio')
    axs[1, 0].set_xlabel('Alpha Values')
    axs[1, 0].set_ylabel('Ratio')

    axs[1, 1].set_title('Largest Connected Component Ratio')
    axs[1, 1].set_xlabel('Alpha Values')
    axs[1, 1].set_ylabel('LCC Size Ratio')

    plt.tight_layout()

    quarters_str = "_".join(map(str, quarters))
    filename = f'network_analysis_{year}_Q{"_".join(map(str, quarters))}_with_normalization.pdf'
    plt.savefig(os.path.join('figs', filename), format='pdf')
    plt.close()

def create_subplot_for_quarter(G, alpha_values, year, quarter):
    original_weight = sum(nx.get_edge_attributes(G, 'weight').values())
    original_nodes = G.number_of_nodes()
    original_lcc_size = largest_component_size(G)
    
    largest_drop = 0
    alpha_before_largest_drop = None
    previous_lcc_ratio = largest_component_size(disparity_filter_alpha_cut(G, alpha_t=alpha_values[0])) / original_lcc_size

    fig = make_subplots(rows=1, cols=4, subplot_titles=["Clustering Coefficient", "Weight Ratio", "Node Ratio", "Largest Component Ratio"])
    
    for i, alpha in enumerate(alpha_values):
        G_filtered = disparity_filter_alpha_cut(G, alpha_t=alpha)
        current_lcc_size = largest_component_size(G_filtered)
        lcc_ratio = current_lcc_size / original_lcc_size

        if i > 0:
            drop = previous_lcc_ratio - lcc_ratio
            if drop > largest_drop:
                largest_drop = drop
                alpha_before_largest_drop = alpha_values[i - 1]
        
        previous_lcc_ratio = lcc_ratio

        current_weight = sum(nx.get_edge_attributes(G_filtered, 'weight').values())
        current_nodes = G_filtered.number_of_nodes()
        current_lcc_size = largest_component_size(G_filtered)

        weight_ratio = current_weight / original_weight
        node_ratio = current_nodes / original_nodes
        lcc_ratio = current_lcc_size / original_lcc_size
        avg_clustering = sum(nx.clustering(G_filtered).values()) / current_nodes

        fig.add_trace(go.Scatter(x=[alpha], y=[avg_clustering], mode='markers', name=f'α={alpha}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[alpha], y=[weight_ratio], mode='markers', name=f'α={alpha}'), row=1, col=2)
        fig.add_trace(go.Scatter(x=[alpha], y=[node_ratio], mode='markers', name=f'α={alpha}'), row=1, col=3)
        fig.add_trace(go.Scatter(x=[alpha], y=[lcc_ratio], mode='markers', name=f'α={alpha}'), row=1, col=4)

    fig.update_layout(height=600, width=2400, title_text=f"Year: {year}, Quarter: {quarter} Alpha before largest LCC drop: {alpha_before_largest_drop}")
    return fig

def main():
    parser = argparse.ArgumentParser(description='Collaboration Network Filter')
    parser.add_argument('--inputFilePath', required=True, help='Location of input Collaboration Network Data edgelist')
    parser.add_argument('--year', type=int, choices=[2020, 2021, 2022, 2023], required=True, help='Year to filter data')
    parser.add_argument('--quarters', type=int, nargs='+', choices=[1, 2, 3, 4], required=True, help='Quarters to filter data')
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

    # Plot all alpha values between 0.5 and 0.01 at 0.01 decrements
    alpha_values = [round(a, 2) for a in np.arange(0.5, 0.009, -0.01)]
    analyze_network(G, alpha_values, args.year, args.quarters)

if __name__ == '__main__':
    main()
