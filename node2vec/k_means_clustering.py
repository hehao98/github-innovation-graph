# Import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from itertools import product


# ============================================================================
# Configuration - ALL 25 combinations with K-means
# ============================================================================

# Base paths
EMBEDDINGS_BASE_PATH = "emb/no_US_experiments/"
MAPPING_PATH = "country_mapping/country_code_to_id_mapping.csv"
OUTPUT_PATH = "KMeans_Community_Results/"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Relevant p and q combinations (4 total) (can also modify to include all p q combinations if interested)
P_VALUES = [0.25, 4]
Q_VALUES = [0.25, 4]

# Community targets 
CLUSTER_TARGETS = [6]

# Generate all configurations
CONFIGS = []
for p in P_VALUES:
    for q in Q_VALUES:
        for n_clusters in CLUSTER_TARGETS:
            # Determine the type based on p,q values
            if p < 1 and q > 1:
                type_name = "structural_equiv"
                description = f"Structural Equivalence (p={p}, q={q})"
            elif p > 1 and q < 1:
                type_name = "homophily"
                description = f"Homophily (p={p}, q={q})"
            elif p == q:
                type_name = "balanced"
                description = f"Balanced (p={p}, q={q})"
            elif p > q:
                type_name = "homophily_leaning"
                description = f"Homophily-leaning (p={p}, q={q})"
            else:
                type_name = "structural_leaning"
                description = f"Structural-leaning (p={p}, q={q})"

            config = {
                "p": p,
                "q": q,
                "n_clusters": n_clusters,
                "type": type_name,
                "name": f"p_{p}_q_{q}_{n_clusters}clusters",
                "description": f"{description} - {n_clusters} clusters"
            }
            CONFIGS.append(config)

print(f"Total configurations to process: {len(CONFIGS)}")
print(f"Embedding files: {len(P_VALUES) * len(Q_VALUES)}")
print(f"Cluster settings: {len(CLUSTER_TARGETS)} (6 clusters)")

# ============================================================================
# Helper Functions
# ============================================================================

def load_embeddings(p, q):
    """Load node2vec embeddings from .emd file"""
    filename = f"country_collab_no_US_p_{p}_q_{q}.emd"
    filepath = os.path.join(EMBEDDINGS_BASE_PATH, filename)

    print(f"  Loading: {filename}")

    embeddings = {}
    try:
        with open(filepath, 'r') as f:
            header = f.readline().strip().split()
            vocab_size, dimensions = int(header[0]), int(header[1])

            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                embedding = np.array([float(x) for x in parts[1:]])
                embeddings[node_id] = embedding

        print(f"    ✓ Loaded embeddings for {len(embeddings)} nodes ({dimensions}D)")
        return embeddings

    except FileNotFoundError:
        print(f"    ❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"    ❌ Error loading {filename}: {e}")
        return None

def load_country_mapping():
    """Load mapping from node ID to country code"""
    print(f"Loading country mapping from: {MAPPING_PATH}")
    df = pd.read_csv(MAPPING_PATH)
    df.columns = df.columns.str.strip()

    id_to_country = dict(zip(df['NodeID'], df['CountryCode']))
    print(f"Loaded mapping for {len(id_to_country)} countries")
    return id_to_country

def run_kmeans_clustering(embeddings, n_clusters, standardize=True):
    """
    Run K-means clustering on embeddings
    Returns cluster assignments and additional metrics
    """
    node_ids = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[node_id] for node_id in node_ids])

    print(f"  Running K-means: {len(node_ids)} nodes, {embedding_matrix.shape[1]}D → {n_clusters} clusters")

    # Optionally standardize embeddings
    if standardize:
        scaler = StandardScaler()
        embedding_matrix = scaler.fit_transform(embedding_matrix)
        print(f"    ✓ Standardized embeddings")

    # Run K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )

    cluster_labels = kmeans.fit_predict(embedding_matrix)

    # Create results dictionary
    results = {
        'node_ids': node_ids,
        'cluster_labels': cluster_labels,
        'inertia': kmeans.inertia_,
        'n_iter': kmeans.n_iter_,
        'cluster_centers': kmeans.cluster_centers_
    }

    # Calculate cluster statistics
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, counts))

    print(f"    ✓ Converged in {kmeans.n_iter_} iterations")
    print(f"    ✓ Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    print(f"    ✓ Cluster sizes: {dict(sorted(cluster_sizes.items()))}")

    return results

def create_output_dataframe(results, id_to_country, column_name):
    """Create output DataFrame with country codes and cluster assignments"""
    output_data = []

    for i, node_id in enumerate(results['node_ids']):
        if node_id in id_to_country:
            country_code = id_to_country[node_id]
            cluster_id = results['cluster_labels'][i]
            output_data.append({
                'Id': country_code,
                column_name: cluster_id
            })

    df = pd.DataFrame(output_data)
    df = df.sort_values('Id')

    print(f"    ✓ Created output with {len(df)} countries")
    return df

def analyze_cluster_composition(df, cluster_column, id_to_country_full):
    """Analyze what countries are in each cluster"""
    print(f"\n    📊 Cluster composition:")

    for cluster_id in sorted(df[cluster_column].unique()):
        countries = df[df[cluster_column] == cluster_id]['Id'].tolist()
        print(f"      Cluster {cluster_id}: {len(countries)} countries")

        # Show sample countries
        sample_size = min(8, len(countries))
        sample_countries = countries[:sample_size]
        if len(countries) > sample_size:
            sample_countries.append(f"... (+{len(countries)-sample_size} more)")
        print(f"        {', '.join(sample_countries)}")

def create_master_summary():
    """Create a master summary of all configurations"""
    summary_data = []

    for config in CONFIGS:
        summary_data.append({
            'p': config['p'],
            'q': config['q'],
            'n_clusters': config['n_clusters'],
            'type': config['type'],
            'filename': f"kmeans_{config['name']}_communities.csv",
            'description': config['description']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_PATH, "KMEANS_MASTER_SUMMARY.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n📋 Master summary saved: KMEANS_MASTER_SUMMARY.csv")
    return summary_df

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("="*80)
    print("K-MEANS COMMUNITY ANALYSIS - ALL 25 EMBEDDING FILES")
    print("="*80)
    print(f"Processing {len(P_VALUES)} × {len(Q_VALUES)} = {len(P_VALUES) * len(Q_VALUES)} embedding files")
    print(f"With {len(CLUSTER_TARGETS)} cluster targets each = {len(CONFIGS)} total configurations")
    print("✅ GUARANTEED: Exactly 4 and 7 clusters for each p,q combination")
    print("="*80)

    # Load country mapping once
    id_to_country = load_country_mapping()

    # Track which embeddings we've loaded (to avoid reloading)
    loaded_embeddings = {}

    # Track results
    successful_configs = 0
    failed_configs = 0
    all_results = []

    # Process each configuration
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n{'─'*70}")
        print(f"[{i:2d}/{len(CONFIGS)}] {config['description']}")
        print(f"{'─'*70}")

        try:
            # Load embeddings (cache to avoid reloading same file)
            embedding_key = (config['p'], config['q'])
            if embedding_key not in loaded_embeddings:
                embeddings = load_embeddings(config['p'], config['q'])
                if embeddings is None:
                    failed_configs += 1
                    continue
                loaded_embeddings[embedding_key] = embeddings
            else:
                print(f"  Using cached embeddings for p={config['p']}, q={config['q']}")

            embeddings = loaded_embeddings[embedding_key]

            # Run K-means clustering
            results = run_kmeans_clustering(embeddings, config['n_clusters'])

            # Create output DataFrame
            column_name = f"kmeans_{config['name']}_communities"
            df_output = create_output_dataframe(results, id_to_country, column_name)

            # Analyze cluster composition
            analyze_cluster_composition(df_output, column_name, id_to_country)

            # Save results
            output_filename = f"kmeans_{config['name']}_communities.csv"
            output_filepath = os.path.join(OUTPUT_PATH, output_filename)
            df_output.to_csv(output_filepath, index=False)

            # Store results for summary
            all_results.append({
                'config': config,
                'inertia': results['inertia'],
                'n_iter': results['n_iter'],
                'filename': output_filename
            })

            # Report success
            print(f"  ✅ Success: EXACTLY {config['n_clusters']} clusters (guaranteed!)")
            print(f"     Saved: {output_filename}")
            successful_configs += 1

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            failed_configs += 1

    # Create master summary
    print(f"\n{'='*80}")
    print("CREATING MASTER SUMMARY")
    print(f"{'='*80}")
    summary_df = create_master_summary()

    # Final report
    print(f"\n{'='*80}")
    print("🎉 K-MEANS ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Successful configurations: {successful_configs}")
    print(f"❌ Failed configurations: {failed_configs}")
    print(f"📁 Results saved to: {OUTPUT_PATH}")
    print(f"📊 Total CSV files created: {successful_configs + 1} (including summary)")

    print(f"\n🎯 Files ready for Gephi:")
    print(f"   • {successful_configs} community assignment files")
    print(f"   • 1 master summary file")

    print(f"\n📋 Parameter combinations processed:")
    print(f"   • P values: {P_VALUES}")
    print(f"   • Q values: {Q_VALUES}")
    print(f"   • Cluster targets: {CLUSTER_TARGETS}")

    # Show summary of types
    type_counts = summary_df['type'].value_counts()
    print(f"\n📊 Configuration types:")
    for config_type, count in type_counts.items():
        print(f"   • {config_type}: {count} configurations")

# ============================================================================
# Run Analysis
# ============================================================================

if __name__ == "__main__":
    main()