import itertools
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


def reciprocity_pho(g: nx.DiGraph, weight: str = "weight", null_model="WCM"):
    """
    The definition comes from:
        Squartini, Tiziano, et al. "Reciprocity of weighted networks."
        Scientific reports 3.1 (2013): 2729.
    """
    pass


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


def reciprocity_analysis(year: int, quarter: int, partition_csv_path: str):
    # NOTE: partition labels are hardcoded and must be changed w.r.t. blockmodeling results
    PARTITION_LABELS = {
        0: "Peripheral",
        1: "Core",
        2: "Semi-Peripheral",
        3: "US",
    }

    metrics = defaultdict(dict)
    g = get_network(year, quarter)
    countries = set(g.nodes())
    p_df = pd.read_csv(partition_csv_path, keep_default_na=False, na_values=[""])

    BM = nx.quotient_graph(
        g,
        partition=[
            list(p_df[(p_df.partition == p) & p_df.country.isin(countries)].country)
            for p in sorted(set(p_df.partition))
        ],
        relabel=True,
        create_using=nx.DiGraph,
        node_data=lambda p: {
            "countries": sorted(p),
        },
        edge_data=lambda p1, p2: {
            "weight": sum(
                g.edges[ctry1, ctry2]["weight"]
                for ctry1 in p1
                for ctry2 in p2
                if g.has_edge(ctry1, ctry2)
            )
        },
    )

    metrics["Reciprocity (r)"] = reciprocity_all(g)
    for p in sorted(set(p_df.partition)):
        label = PARTITION_LABELS[p]

        total_within_weight = 0
        for x, y in itertools.product(p_df[p_df.partition == p].country, repeat=2):
            if g.has_edge(x, y):
                total_within_weight += g.edges[x, y]["weight"]

        metrics[f"Total Weight Inbound"][label] = BM.in_degree(p, weight="weight")
        metrics[f"Total Weight Outbound"][label] = BM.out_degree(p, weight="weight")
        metrics[f"Total Weight Within"][label] = total_within_weight

    return metrics


def plot_reciprocity(year_quarters, metrics):
    year_quarters = [f"{x[0]} Q{x[1]}" for x in year_quarters]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(
        range(len(year_quarters)), [x["Reciprocity (r)"] for x in metrics], marker="o"
    )
    ax.set_xticks(range(len(year_quarters)))
    ax.set_xticklabels(year_quarters)
    ax.set_xlabel("Year-Quarter")
    ax.set_ylabel("Overall Reciprocity (r)")
    fig.autofmt_xdate()
    fig.savefig("figs/reciprocity_r.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for i, label in enumerate(metrics[0]["Total Weight Inbound"].keys()):
        inbound = [x["Total Weight Inbound"][label] for x in metrics]
        axes[i].plot(
            range(len(year_quarters)),
            inbound,
            marker="o",
            label=f"Inbound ({label})",
        )

        outbound = [x["Total Weight Outbound"][label] for x in metrics]
        axes[i].plot(
            range(len(year_quarters)),
            outbound,
            marker="x",
            label=f"Outbound ({label})",
        )
        
        within = [x["Total Weight Within"][label] for x in metrics]
        axes[i].plot(
            range(len(year_quarters)),
            within,
            marker="^",
            label=f"Within ({label})",
        )

        axes[i].set_xticks(range(len(year_quarters)))
        axes[i].set_xticklabels(year_quarters)
        axes[i].set_xlabel("Year-Quarter")
        axes[i].set_ylabel("Total Weight")
        axes[i].set_ylim(0)
        axes[i].legend()
    fig.autofmt_xdate()
    fig.savefig("figs/reciprocity_in_out_weight.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    graph_csv_path = "data/economy_collaborators.csv"
    partition_csv_path = "data/blockmodeling_partitions.csv"
    graph_data = pd.read_csv(graph_csv_path, keep_default_na=False, na_values=[""])
    year_quarters = sorted(set(zip(graph_data["year"], graph_data["quarter"])))

    # block modling in all quarters
    block_modeling(partition_csv_path=partition_csv_path)

    # reciprocity analysis in each quarter
    metrics = [reciprocity_analysis(y, q, partition_csv_path) for y, q in year_quarters]
    plot_reciprocity(year_quarters, metrics)


if __name__ == "__main__":
    main()
