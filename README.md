# Replication Package: The Structure of Cross-National Collaboration in Open-Source Software Development

This repository contains the replication package for the following paper:

> Henry Xu, Katy Yu, Hao He, Hongbo Fang, Bogdan Vasilescu, and Patrick S. Park. 2025. The Structure of Cross-National Collaboration in Open-Source Software Development. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM ’25), November 10–14, 2025, Seoul, Republic of Korea. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3746252.3761237

## Environment Requirements

This replication package works with Python 3.10+, [Gephi](https://gephi.org/) 0.10+, and the latest version of R. The required Python dependencies are described in [requirements.txt](requirements.txt).

## Dataset 

* [data/economy_collaborators.csv](data/economy_collaborators.csv): GitHub collaboration graph aggregated from [this repository](https://github.com/github/innovationgraph).
* [data/ctry_civ_labels.csv](data/ctry_civ_labels.csv): Country civilization labels as categorized by [Huntington (1987)](http://www.jstor.org/stable/20045621).
* [data/gdp_per_capita.csv](data/gdp_per_capita.csv): Country-level GDP per-capita data collected from the [World Bank](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)

## Data Preprocessing (Section 3.1)

This section provides documentation regarding the disparity filter.

```shell
python edge_filtering/disparity_filter_alpha_plots.py --inputFilePath data/economy_collaborators.csv --year 2023 --quarter 1 --normalize outgoing
```

Flags:

* --inputFilePath economy_collaborators.csv (Original Github innovation graph data for economy_collaborators.csv)
* --year 2023 (year for analysis)
* --quarter 1 (quarter for analysis)
* --normalize outgoing (normalize by sender total weight)

Key metrics:

1. Cumulative Degree Distribution
2. Distribution of Link Weights
3. Clustering coefficients, percent of total weight, percent of total nodes
4. Largest Connected Component ratio (after vs before alpha filtering)

For metrics 3 and 4, the purple vertical line highlights the chosen alpha value based on the Largest Connected Component Ratio (right before the biggest drop is chosen). We used Weakly Connected Component for the Largest Connected Component measurement.

To generate data for visualization:

```shell
python edge_filtering/disparity_filter.py --inputFilePath data/economy_collaborators.csv --outputFilePath data/filtered/economy_collaborators_outgoing --normalize outgoing --excludeCountries EU --optimalAlpha 0.09
```

Flags:

* --inputFilePath economy_collaborators.csv: Use original GitHub innovation graph data for economy_collaborators.csv 
* --outputFilePath: this is for the output filename prefix. We used filtered_graph_test_combine_all_exclude_EU_normalize_sender to represent what we did, which is summing all the edge weights from all years and quarters into one file for each edge, and then we excluded EU, normalized by sender total weights. 
* --normalize outgoing: this flag is for normalizing the weights. The options are outgoing, incoming, log, or none (default). In our case, we normalized by outgoing, which is normalization by total sender country weight across all its edges.
* --excludeCountries: this flag allows for entering country codes to exclude from the data. In our case, we excluded the EU
* --optimalAlpha: this flag allows for entering a list of alpha values for filtering, and the script will generate one CSV output per alpha value. In our case, we used  0.09, 0.12, 0.15, 1 (no filtering) as alpha values.

## Hierarchical Clustering (Section 3.2)

The following two Python files replicate Figure 1 and Figure 2.

```shell
python blockmodeling.py
python reciprocity.py
```

## Exponential Random Graph Models (Section 3.3)

The ERGM results were obtained from the following [R file](network_analysis/github_civilization.R). 

## Node2Vec (Section 3.4)

The Node2Vec results can be obtained by following these 3 steps:
1. Embedding generation:
Download all files in github-innovation-graph/node2vec/
    Including the edgelist from graph and the two python files     
    1. graph/economy_collaborators_no_US_with_weights.edgelist
    2. src/node2vec.py
    3. src/main.py

Note: It is important that you then go into the node2vec folder (if you're running this on terminal, ensure you are in the node2vec folder, otherwise you need to update the paths to match)

Install requirements for node2vec with node2vec/node2vec_requirements.txt

```shell
pip install -r node2vec_requirements.txt
```

then run the command: 
```shell
python node2vec_no_us.py
```
This will generate all 25 embedding files in the node2vec folder (emb/no_US_experiments/)
    
2. Silhouette score:

run the command: 
```shell
python silhouette_analysis.py
```
This will generate elbow and silhouette analysis and show some plots. We will be using the 6 clusters from the silhouette analysis for both homophily and structural equivalence case. The homophily setting (p = 4, q = 0.25), silhouette score was highest at 3 clusters, and then 6, but 3 clusters was merging too many countries into one cluster (and thus not as informative), so we used 6 clusters. Similarly, for structural equivalence case (p = 0.25, q = 4), the silhouette score was highest at 2 clusters, and then 6, but 2 clusters was too few clusters to be informative.  

3. K-means clustering based on 6 clusters from silhouette analysis: 

run the command: 
```shell
python k_means_clustering.py
```

This will generate the csv files for k-means clustering based on 6 clusters from silhouette analysis. The csv files will be saved in the KMeans_Community_Results folder. You can then take these csv files and import them into Gephi to visualize the communities. 

Note that the current settings only have 2 p and q values (0.25 and 4) and 1 cluster target (6 clusters). You can modify the P_VALUES, Q_VALUES, and CLUSTER_TARGETS variables in the k_means_clustering.py file to change the settings.