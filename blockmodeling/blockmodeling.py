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


def get_network(year: int = None, quarter: int = None):
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


def block_modeling(year: int = None, quarter: int = None, partition_csv_path: str = None):
    g = get_network(year, quarter)
    partitions, threshold, score_records = get_optimal_hc(g, 20, 150, 1)

    if partition_csv_path is not None:
        with open(partition_csv_path, "w") as f:
            f.write("country,partition\n")
            for i, partition in enumerate(partitions):
                for ctry in partition:
                    f.write(f"{ctry},{i}\n")

    if year is not None and quarter is not None:
        suffix = f"{year}_q{quarter}"
        print(f"{year} Q{quarter}:")
    else:
        suffix = "all"
        print(f"Across all quarters:")

    fig, ctry_to_color = plot_dendrogram(g, t=threshold)
    fig.savefig(f"figs/blockmodeling_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plot_score_selection(threshold, score_records)
    fig.savefig(f"figs/blockmodeling_scores_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    gdp_data = pd.read_csv("data/gdp_per_capita.csv", keep_default_na=False, na_values=[""])
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
    ax.set_ylabel("GDP per Capita")
    ax.set_xlabel("Country Partitiion")
    fig.savefig(f"figs/blockmodeling_gdp_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_reciprocity(year_quarters, metrics):
    year_quarters = [f"{x[0]} Q{x[1]}" for x in year_quarters]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(range(len(year_quarters)), [x["reciprocity"] for x in metrics], marker="o")
    ax.set_xticks(range(len(year_quarters)))
    ax.set_xticklabels(year_quarters)
    ax.set_xlabel("Year-Quarter")
    ax.set_ylabel("Overall Reciprocity")
    fig.autofmt_xdate()
    fig.savefig("figs/reciprocity.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for label in ["core", "semi", "periphery"]:
        non_recipro_out = [x[f"reciprocity_{label}"][0] for x in metrics]
        if not all(x == 0 for x in non_recipro_out):
            ax.plot(
                range(len(year_quarters)),
                non_recipro_out,
                marker="o",
                label=f"Non-Reciprocated Out ({label})",
            )
        non_recipro_in = [x[f"reciprocity_{label}"][1] for x in metrics]
        if not all(x == 0 for x in non_recipro_in):
            ax.plot(
                range(len(year_quarters)),
                non_recipro_in,
                marker="o",
                label=f"Non-Reciprocated In ({label})",
            )
    ax.set_xticks(range(len(year_quarters)))
    ax.set_xticklabels(year_quarters)
    ax.set_xlabel("Year-Quarter")
    ax.set_ylabel("Reciprocated Weight")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig("figs/reciprocity_node.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for label in [
        "reciprocity_core_semi",
        "reciprocity_core_periphery",
        "reciprocity_semi_periphery",
    ]:
        ax.plot(
            range(len(year_quarters)),
            [x[label] for x in metrics],
            marker="o",
            label=label.replace("reciprocity", "").replace("_", " ").title(),
        )
    ax.set_xticks(range(len(year_quarters)))
    ax.set_xticklabels(year_quarters)
    ax.set_xlabel("Year-Quarter")
    ax.set_ylabel("Non-Reciprocated Inbound Weight")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig("figs/reciprocity_dyad.pdf", bbox_inches="tight")
    plt.close(fig)



def reciprocity_all(g: nx.DiGraph, weight: str = "weight"):
    """
    The definition comes from:
        Squartini, Tiziano, et al. "Reciprocity of weighted networks."
        Scientific reports 3.1 (2013): 2729.
    """
    total, total_reciprocal = 0, 0
    for n1 in g.nodes():
        for n2 in g.nodes():
            if n1 == n2:
                continue
            w1 = g.get_edge_data(n1, n2, default={weight: 0})[weight]
            w2 = g.get_edge_data(n2, n1, default={weight: 0})[weight]
            total_reciprocal += min(w1, w2)
            total += w1
    return total_reciprocal / total


def reciprocity_node(g: nx.DiGraph, node: str, weight: str = "weight"):
    reciprocated_strength = 0
    total_out, total_in = 0, 0
    for n2 in g.nodes():
        if node == n2:
            continue
        w1 = g.get_edge_data(node, n2, default={weight: 0})[weight]
        w2 = g.get_edge_data(n2, node, default={weight: 0})[weight]
        reciprocated_strength += min(w1, w2)
        total_out += w1
        total_in += w2
    return total_out - reciprocated_strength, total_in - reciprocated_strength


def reciprocity_dyad(g: nx.DiGraph, node1: str, node2: str, weight: str = "weight"):
    w1 = g.get_edge_data(node1, node2, default={weight: 0})[weight]
    w2 = g.get_edge_data(node2, node1, default={weight: 0})[weight]
    if w1 > w2:
        return min(w1, w2) - w1
    else:
        return w2 - min(w1, w2)


def main():
    file_path = "data/economy_collaborators.csv"
    data = pd.read_csv(file_path, keep_default_na=False, na_values=[""])
    year_quarters = sorted(set(zip(data["year"], data["quarter"])))

    # all quarters
    block_modeling(partition_csv_path="data/blockmodeling_partitions.csv")

    # TODO: reciprocity


if __name__ == "__main__":
    main()
