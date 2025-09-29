# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Base paths
EMBEDDINGS_BASE_PATH = "emb/no_US_experiments/"
MAPPING_PATH = "country_mapping/country_code_to_id_mapping.csv"
OUTPUT_PATH = "Optimal_Clustering_Analysis/"

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Key configurations to analyze in detail
KEY_CONFIGS = [
    {"p": 4, "q": 0.25, "name": "strong_homophily", "description": "Strong Homophily (p=4, q=0.25)"},
    {"p": 4, "q": 0.5, "name": "moderate_homophily", "description": "Moderate Homophily (p=4, q=0.5)"},
    {"p": 0.25, "q": 4, "name": "structural_equiv", "description": "Structural Equivalence (p=0.25, q=4)"},
    {"p": 1, "q": 1, "name": "balanced", "description": "Balanced (p=1, q=1)"},
    {"p": 2, "q": 0.5, "name": "homophily_leaning", "description": "Homophily-leaning (p=2, q=0.5)"}
]

# Range of clusters to test
CLUSTER_RANGE = range(2, 12)  # Test 2 to 11 clusters

# ============================================================================
# Helper Functions
# ============================================================================

def load_embeddings(p, q):
    """Load node2vec embeddings from .emd file"""
    filename = f"country_collab_no_US_p_{p}_q_{q}.emd"
    filepath = os.path.join(EMBEDDINGS_BASE_PATH, filename)

    embeddings = {}
    with open(filepath, 'r') as f:
        header = f.readline().strip().split()
        vocab_size, dimensions = int(header[0]), int(header[1])

        for line in f:
            parts = line.strip().split()
            node_id = int(parts[0])
            embedding = np.array([float(x) for x in parts[1:]])
            embeddings[node_id] = embedding

    return embeddings

def load_country_mapping():
    """Load mapping from node ID to country code"""
    df = pd.read_csv(MAPPING_PATH)
    df.columns = df.columns.str.strip()
    return dict(zip(df['NodeID'], df['CountryCode']))

def compute_clustering_metrics(embedding_matrix, cluster_range):
    """Compute elbow (inertia) and silhouette scores for different cluster numbers"""
    inertias = []
    silhouette_scores = []

    for n_clusters in cluster_range:
        # Run K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding_matrix)

        # Compute metrics
        inertia = kmeans.inertia_
        sil_score = silhouette_score(embedding_matrix, cluster_labels)

        inertias.append(inertia)
        silhouette_scores.append(sil_score)

        print(f"  n_clusters={n_clusters:2d}: inertia={inertia:8.1f}, silhouette={sil_score:.3f}")

    return inertias, silhouette_scores

def find_optimal_clusters(cluster_range, inertias, silhouette_scores):
    """Find optimal number of clusters using elbow method and silhouette score"""

    # Method 1: Elbow method (find knee point)
    # Calculate second derivative to find elbow
    if len(inertias) >= 3:
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)

        elbow_idx = np.argmax(second_derivatives) + 1  # +1 because we start from index 1
        elbow_clusters = list(cluster_range)[elbow_idx]
    else:
        elbow_clusters = list(cluster_range)[0]

    # Method 2: Best silhouette score
    best_sil_idx = np.argmax(silhouette_scores)
    best_sil_clusters = list(cluster_range)[best_sil_idx]

    return elbow_clusters, best_sil_clusters

def plot_clustering_metrics(cluster_range, inertias, silhouette_scores, config, output_path):
    """Plot elbow curve and silhouette scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Elbow plot
    ax1.plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title(f'Elbow Method\n{config["description"]}')
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    ax2.plot(cluster_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title(f'Silhouette Analysis\n{config["description"]}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_path, f"{config['name']}_clustering_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    return plot_path

def create_tsne_visualization(embedding_matrix, node_ids, id_to_country, optimal_clusters, config, output_path):
    """Create t-SNE visualization with optimal clustering"""
    print(f"  Creating t-SNE visualization...")

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_coords = tsne.fit_transform(embedding_matrix)

    # Run K-means with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding_matrix)

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Create color map
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_clusters))

    # Plot each cluster
    for cluster_id in range(optimal_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.7, s=60)

    # Add country labels
    for i, node_id in enumerate(node_ids):
        country_code = id_to_country.get(node_id, str(node_id))
        plt.annotate(country_code, (tsne_coords[i, 0], tsne_coords[i, 1]),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=8, alpha=0.8)

    plt.title(f't-SNE Visualization with {optimal_clusters} Clusters\n{config["description"]}',
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Save plot
    plot_path = os.path.join(output_path, f"{config['name']}_tsne_{optimal_clusters}clusters.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    return cluster_labels, plot_path

def analyze_cluster_composition(cluster_labels, node_ids, id_to_country, config):
    """Analyze the composition of each cluster"""
    print(f"\n  📊 Cluster Composition Analysis:")

    cluster_analysis = {}
    for cluster_id in range(len(np.unique(cluster_labels))):
        mask = cluster_labels == cluster_id
        cluster_countries = [id_to_country.get(node_ids[i], str(node_ids[i]))
                           for i in range(len(node_ids)) if mask[i]]

        cluster_analysis[cluster_id] = cluster_countries

        print(f"    Cluster {cluster_id}: {len(cluster_countries)} countries")
        sample_countries = cluster_countries[:8]
        if len(cluster_countries) > 8:
            sample_countries.append(f"... (+{len(cluster_countries)-8} more)")
        print(f"      {', '.join(sample_countries)}")

    return cluster_analysis

def save_optimal_clustering_results(cluster_labels, node_ids, id_to_country, config, optimal_clusters, output_path):
    """Save the optimal clustering results as CSV"""
    results = []
    for i, node_id in enumerate(node_ids):
        if node_id in id_to_country:
            results.append({
                'Id': id_to_country[node_id],
                f'{config["name"]}_optimal_{optimal_clusters}clusters': cluster_labels[i]
            })

    df = pd.DataFrame(results).sort_values('Id')

    csv_path = os.path.join(output_path, f"{config['name']}_optimal_{optimal_clusters}clusters.csv")
    df.to_csv(csv_path, index=False)

    print(f"  💾 Saved optimal clustering: {config['name']}_optimal_{optimal_clusters}clusters.csv")
    return csv_path

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("="*80)
    print("OPTIMAL CLUSTERING ANALYSIS - DATA-DRIVEN CLUSTER SELECTION")
    print("="*80)
    print("🔍 Using elbow method, silhouette analysis, and t-SNE visualization")
    print("📊 Finding optimal number of clusters for key parameter combinations")
    print("="*80)

    # Load country mapping
    id_to_country = load_country_mapping()

    # Track results
    all_results = []

    # Analyze each key configuration
    for i, config in enumerate(KEY_CONFIGS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(KEY_CONFIGS)}] ANALYZING: {config['description']}")
        print(f"{'='*60}")

        try:
            # Load embeddings
            print(f"Loading embeddings for p={config['p']}, q={config['q']}...")
            embeddings = load_embeddings(config['p'], config['q'])

            # Prepare data
            node_ids = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[node_id] for node_id in node_ids])

            # Standardize embeddings
            scaler = StandardScaler()
            embedding_matrix = scaler.fit_transform(embedding_matrix)

            print(f"Prepared {len(node_ids)} countries with {embedding_matrix.shape[1]}D embeddings")

            # Compute clustering metrics
            print(f"Computing clustering metrics for {len(CLUSTER_RANGE)} different cluster counts...")
            inertias, silhouette_scores = compute_clustering_metrics(embedding_matrix, CLUSTER_RANGE)

            # Find optimal clusters
            elbow_clusters, best_sil_clusters = find_optimal_clusters(CLUSTER_RANGE, inertias, silhouette_scores)

            print(f"\n  🎯 Optimal clusters:")
            print(f"     Elbow method: {elbow_clusters} clusters")
            print(f"     Best silhouette: {best_sil_clusters} clusters")

            # Choose the optimal (prefer silhouette score)
            optimal_clusters = best_sil_clusters
            print(f"     SELECTED: {optimal_clusters} clusters (based on silhouette score)")

            # Plot clustering metrics
            print(f"\n  📈 Creating clustering metrics plots...")
            plot_path = plot_clustering_metrics(CLUSTER_RANGE, inertias, silhouette_scores, config, OUTPUT_PATH)

            # Create t-SNE visualization
            print(f"  🎨 Creating t-SNE visualization...")
            cluster_labels, tsne_path = create_tsne_visualization(
                embedding_matrix, node_ids, id_to_country, optimal_clusters, config, OUTPUT_PATH)

            # Analyze cluster composition
            cluster_analysis = analyze_cluster_composition(cluster_labels, node_ids, id_to_country, config)

            # Save results
            csv_path = save_optimal_clustering_results(
                cluster_labels, node_ids, id_to_country, config, optimal_clusters, OUTPUT_PATH)

            # Store results summary
            all_results.append({
                'config': config,
                'elbow_clusters': elbow_clusters,
                'silhouette_clusters': best_sil_clusters,
                'optimal_clusters': optimal_clusters,
                'best_silhouette_score': max(silhouette_scores),
                'cluster_analysis': cluster_analysis,
                'plot_path': plot_path,
                'tsne_path': tsne_path,
                'csv_path': csv_path
            })

            print(f"  ✅ Analysis complete for {config['name']}")

        except Exception as e:
            print(f"  ❌ Error analyzing {config['name']}: {str(e)}")

    # Create summary report
    print(f"\n{'='*80}")
    print("📋 SUMMARY REPORT")
    print(f"{'='*80}")

    summary_data = []
    for result in all_results:
        config = result['config']
        summary_data.append({
            'Configuration': config['description'],
            'p': config['p'],
            'q': config['q'],
            'Elbow_Clusters': result['elbow_clusters'],
            'Silhouette_Clusters': result['silhouette_clusters'],
            'Optimal_Clusters': result['optimal_clusters'],
            'Best_Silhouette_Score': f"{result['best_silhouette_score']:.3f}",
            'CSV_File': os.path.basename(result['csv_path'])
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_path = os.path.join(OUTPUT_PATH, "OPTIMAL_CLUSTERING_SUMMARY.csv")
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {OUTPUT_PATH}")
    print(f"📊 Files created:")
    print(f"   • {len(all_results)} clustering metric plots")
    print(f"   • {len(all_results)} t-SNE visualizations")
    print(f"   • {len(all_results)} optimal clustering CSV files")
    print(f"   • 1 summary report")

    print(f"\n🔬 Key Insights:")
    for result in all_results:
        config = result['config']
        print(f"   • {config['description']}: {result['optimal_clusters']} clusters optimal")

    print(f"\n🎯 Next Steps:")
    print(f"   1. Review the t-SNE plots to validate cluster quality")
    print(f"   2. Use the optimal CSV files for Gephi visualization")
    print(f"   3. Compare how different p,q values affect optimal cluster count")
    print(f"   4. Focus on configurations with highest silhouette scores")

# ============================================================================
# Run Analysis
# ============================================================================

if __name__ == "__main__":
    main()