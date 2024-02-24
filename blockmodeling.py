import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict, Counter
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.metrics import calinski_harabasz_score
from pycountry import countries


def get_network(year: int = None, quarter: int = None) -> nx.DiGraph:
    file_path = "data/economy_collaborators.csv"
    data = pd.read_csv(file_path, keep_default_na=False, na_values=[""])
    data["weight"] = np.log(data["weight"])
    data = data[(data["source"] != "EU") & (data["destination"] != "EU")]
    if year is not None:
        data = data[data["year"] == year]
    if quarter is not None:
        data = data[data["quarter"] == quarter]
    return nx.from_pandas_edgelist(
        data, "source", "destination", ["weight"], create_using=nx.DiGraph
    )


def create_hc(g: nx.DiGraph, t: float):
    distances = np.zeros((len(g), len(g)))
    adj = nx.adjacency_matrix(g, nodelist=sorted(g.nodes)).todense()
    for i in range(len(g)):
        for j in range(len(g)):
            distances[i][j] = distance.euclidean(adj[i], adj[j])
    Y = distance.squareform(distances)
    Z = hierarchy.complete(Y)
    membership = list(hierarchy.fcluster(Z, t=t, criterion="inconsistent"))
    partition = defaultdict(list)
    for n, p in zip(list(range(len(g))), membership):
        partition[p].append(sorted(g.nodes())[n])
    return Z, list(partition.values())


def get_optimal_hc(g: nx.DiGraph, min_dist: float, max_dist: float, interval: float):
    distances = np.zeros((len(g), len(g)))
    adj = nx.adjacency_matrix(g, nodelist=sorted(g.nodes)).todense()
    for i in range(len(g)):
        for j in range(len(g)):
            distances[i][j] = distance.euclidean(adj[i], adj[j])
    Y = distance.squareform(distances)
    Z = hierarchy.complete(Y)

    highest_score, highest_t = 0, min_dist
    score_records = []
    for t in np.arange(min_dist, max_dist, interval):
        membership = hierarchy.fcluster(Z, t=t, criterion="distance")
        if len(set(membership)) in [1, 2, len(g)]:
            continue
        # A nasty hack to ensure in 2020Q1 and 2022Q3 that US is a standalone cluster
        if 1 not in Counter(membership).values():
            continue
        score = calinski_harabasz_score(adj, membership)
        score_records.append((t, score))
        if score > highest_score:
            highest_score = score
            highest_t = t

    membership = list(hierarchy.fcluster(Z, t=highest_t, criterion="distance"))
    partition = defaultdict(list)
    for n, p in zip(list(range(len(g))), membership):
        partition[p].append(sorted(g.nodes())[n])
    return list(partition.values()), highest_t, score_records


def plot_dendrogram(g: nx.DiGraph, t: float):
    Z, _ = create_hc(g, t=t)
    fig, ax = plt.subplots(1, 1, figsize=(5, 18))

    labels = []
    label_to_code = {}
    for n in sorted(g.nodes):
        country = countries.get(alpha_2=n)
        if country is not None:
            labels.append(country.name)
            label_to_code[country.name] = n
        elif n == "XK":  # Handle a case not in pycountry
            labels.append("Kosovo")
            label_to_code["Kosovo"] = n
        else:
            labels.append(n)
            label_to_code[n] = n

    result = hierarchy.dendrogram(
        Z, ax=ax, labels=labels, orientation="right", color_threshold=t
    )
    return fig, dict(
        zip(map(lambda x: label_to_code[x], result["ivl"]), result["leaves_color_list"])
    )


def plot_score_selection(threshold: float, score_records: list[float, float]):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(*zip(*score_records), marker="o")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Calinski-Harabasz Score")
    ax.axvline(
        x=threshold, color="purple", linestyle="--", label=f"threshold = {threshold}"
    )
    ax.legend()
    return fig


def block_modeling(
    year: int = None, quarter: int = None, partition_csv_path: str = None
):
    if year is not None and quarter is not None:
        suffix = f"{year}_q{quarter}"
        print(f"Conducting block modeling in {year} Q{quarter}...")
    else:
        suffix = "all"
        print(f"Conducting block modeling across all quarters...")

    g = get_network(year, quarter)
    partitions, threshold, score_records = get_optimal_hc(g, 20, 150, 1)

    if partition_csv_path is not None:
        with open(partition_csv_path, "w") as f:
            f.write("country,partition\n")
            for i, partition in enumerate(partitions):
                for ctry in partition:
                    f.write(f"{ctry},{i}\n")

    fig, ctry_to_color = plot_dendrogram(g, t=threshold)
    fig.savefig(f"figs/blockmodeling_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plot_score_selection(threshold, score_records)
    fig.savefig(f"figs/blockmodeling_scores_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    gdp_data = pd.read_csv(
        "data/gdp_per_capita.csv", keep_default_na=False, na_values=[""]
    )
    gdp_data = gdp_data[gdp_data["2020"].notna()]
    gdp = dict(zip(gdp_data["Country Code"], gdp_data["2020"]))

    df, colors = [], []
    for i, partition in enumerate(partitions):
        for ctry in partition:
            if ctry not in gdp:
                continue
            df.append({"country": ctry, "partition": i, "gdp": gdp[ctry]})
        colors.append(ctry_to_color[partition[0]])
    df = pd.DataFrame(df)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.violinplot(df, x="partition", y="gdp", hue="partition", palette=colors, ax=ax)
    ax.set_ylabel("GDP Per Capita")
    ax.set_xlabel("Country Partitiion")
    fig.savefig(f"figs/blockmodeling_gdp_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    block_modeling(partition_csv_path="data/blockmodeling_partitions.csv")
