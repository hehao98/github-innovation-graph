# Replication Package: The Structure of Cross-National Collaboration in Open-Source Software Development

## Environment Requirements

This replication package is tested with Python 3.10, [Gephi](https://gephi.org/) 0.10 and R xxx. The required Python dependencies are described in [requirements.txt](requirements.txt).

## Replication Instructions

### Dataset 

* [data/economy_collaborators.csv](data/economy_collaborators.csv): GitHub collaboration graph aggregated from [this repository](https://github.com/github/innovationgraph).
* [data/ctry_civ_labels.csv](data/ctry_civ_labels.csv): Country civilization labels as categorized by [Huntington (1987)](http://www.jstor.org/stable/20045621).
* [data/gdp_per_capita.csv](data/gdp_per_capita.csv): Country-level GDP per-capita data collected from the [World Bank](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)

### Data Preprocessing 

To replicate the alpha value plots in the paper:

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

* --inputFilePath economy_collaborators.csv: Use original github innovation graph data for economy_collaborators.csv 
* --outputFilePath this is for output filename prefix. We used filtered_graph_test_combine_all_exclude_EU_normalize_sender to represent what we did, which is summing all the edge weights from all year and quarters into one file for each edge, and then we excluded EU, normalized by sender total weights. 
* --normalize outgoing: this flag is for normalizing the weights. The options are outgoing, incoming, log, or none (default). In our case, we normalized by outgoing, which is normalization by total sender country weight across all its edges.
* --excludeCountries: this flag allows for entering country codes to exclude from the data. In our case, we excluded EU
* --optimalAlpha: this flag allows for entering a list of alpha values for filtering and the script will generate one csv output per alpha value. In our case, we used  0.09 0.12 0.15 1 (no filtering) as alpha values.

### Political Events

### Cultural Homophily

The Gephi visualization files can be found in the [visualization/](visualization/) folder.

### World Systems

All results can be generated with the following script:

```shell
python blockmodeling.py
python reciprocity.py
```
