library(dplyr)
library(igraph)
library(statnet)
library(intergraph)
library(rstan)
library(tidyverse)
library(stargazer)
library(ergm)
rm(list = ls())

# df = read.csv("~/github-innovation-graph/data/economy_collaborators_network_combine_all_year_quarters_exclude_EU_normalize_sender_only_civilization_countries/economy_collaborators_outgoing_alpha_0_01.csv")
# df = read.csv("~/github-innovation-graph/data/economy_collaborators_network_combine_all_year_quarters_exclude_EU_normalize_sender_only_civilization_countries/economy_collaborators_outgoing_alpha_0_05.csv")
df = read.csv("~/github-innovation-graph/data/economy_collaborators_network_combine_all_year_quarters_exclude_EU_normalize_sender_only_civilization_countries/economy_collaborators_network_combine_all_exclude_EU_normalize_sender_alpha_0_09_valid_countries.csv")

df[, "source"] <- as.character(df[, "source"])
df[, "target"] <- as.character(df[, "target"])

# Create directed graph
df_edge <- graph_from_data_frame(df, directed = TRUE)

# Convert to undirected graph, merging multiple edges
# df_edge <- as.undirected(df_edge, mode = "collapse", edge.attr.comb = list(weight = "sum"))

# Civilization
df_civ = read.csv("~/github-innovation-graph/data/ctry_civ_labels.csv")
df_civ[, "Id"] <- as.character(df_civ[, "Id"])
df_civ[, "civilizations"] <- as.character(df_civ[, "civilizations"])
# Finally, add attributes  
linked_ids_civ <- match(V(df_edge)$name, df_civ$Id)

# if there are missing data in civ, run this and rerun the code block up there
valid_nodes <- intersect(V(df_edge)$name, df_civ$Id)
df_edge <- induced_subgraph(df_edge, vids = valid_nodes)

# Sum Weight
df_weight = read.csv("~/github-innovation-graph/data/node_attributes.csv")
df_weight[, "Node"] <- as.character(df_weight[, "Node"])
# Finally, add attributes  
linked_ids_weight <- match(V(df_edge)$name, df_weight$Node)

# GDP Per Capita
df_gdp = read.csv("~/github-innovation-graph/data/gdp_per_capita.csv")
df_gdp[, "Country.Code"] <- as.character(df_gdp[, "Country.Code"])
# Finally, add attributes  
linked_ids_gdp <- match(V(df_edge)$name, df_gdp$Country.Code)

# Compute Average for GDP
df_gdp$average <- rowMeans(df_gdp[,c('X2020', 'X2021', 'X2022')], na.rm = TRUE)

# Block Model
df_blockmodel = read.csv("~/github-innovation-graph/data/blockmodeling_partitions.csv")
df_blockmodel[, "country"] <- as.character(df_blockmodel[, "country"])
df_blockmodel[, "partition"] <- as.character(df_blockmodel[, "partition"])
# Finally, add attributes  
linked_ids_block <- match(V(df_edge)$name, df_blockmodel$country)

# Invasion / Colonization
df_colonization = read.csv("~/github-innovation-graph/data/COLDAT_colonies_processed.csv")
df_colonization[, "Id"] <- as.character(df_colonization[, "Id"])
# Finally, add attributes  
linked_ids_colonization <- match(V(df_edge)$name, df_colonization$Id)

# View edge attributes of graph object
df_edge
edge_attr(df_edge)
V(df_edge)$name
df_civ$civilizations[linked_ids_civ]
df_weight$LogSumWeightOut[linked_ids_weight]
df_weight$LogSumWeightIn[linked_ids_weight]
df_blockmodel$partition[linked_ids_block]
df_colonization$col.belgium[linked_ids_colonization]
df_colonization$col.italy[linked_ids_colonization]
df_colonization$col.france[linked_ids_colonization]
df_colonization$col.britain[linked_ids_colonization]
df_colonization$col.USSR[linked_ids_colonization]
df_colonization$col.portugal[linked_ids_colonization]
df_colonization$col.spain[linked_ids_colonization]
df_colonization$col.japan[linked_ids_colonization]

V(df_edge)$civilizations <- df_civ$civilizations[linked_ids_civ]
V(df_edge)$LogSumWeightOut <- df_weight$LogSumWeightOut[linked_ids_weight]
V(df_edge)$LogSumWeightIn <- df_weight$LogSumWeightIn[linked_ids_weight]
V(df_edge)$GDP <- df_gdp$average[linked_ids_gdp]
V(df_edge)$blockmodel <- df_blockmodel$partition[linked_ids_block]

V(df_edge)$col.belgium <- df_colonization$col.belgium[linked_ids_colonization]
V(df_edge)$col.britain <- df_colonization$col.britain[linked_ids_colonization]
V(df_edge)$col.france <- df_colonization$col.france[linked_ids_colonization]
V(df_edge)$col.germany <- df_colonization$col.germany[linked_ids_colonization]
V(df_edge)$col.italy <- df_colonization$col.italy[linked_ids_colonization]
V(df_edge)$col.netherlands <- df_colonization$col.netherlands[linked_ids_colonization]
V(df_edge)$col.portugal <- df_colonization$col.portugal[linked_ids_colonization]
V(df_edge)$col.spain <- df_colonization$col.spain[linked_ids_colonization]
V(df_edge)$col.japan <- df_colonization$col.japan[linked_ids_colonization]
V(df_edge)$col.USSR <- df_colonization$col.USSR[linked_ids_colonization]


g = as_edgelist(df_edge)
assortativity(df_edge, values = V(df_edge)$civilization)

# ERGM

df_edge_net = asNetwork(df_edge)

network.size(df_edge_net)
df_edge_net %v% 'civilizations'
get.vertex.attribute(df_edge_net, 'civilizations')
network.density(df_edge_net)

formula <- df_edge_net ~ edges +
  nodematch("civilizations", diff = TRUE) +
  # nodematch("blockmodel", diff = FALSE) + 
  mutual
  # nodematch("blockmodel", diff = FALSE)
  # nodematch("col.britain", diff = TRUE) +
  # nodematch("col.france", diff = TRUE) +
  # nodematch("col.germany", diff = TRUE) +
  # nodematch("col.USSR", diff = TRUE) +
  # nodematch("col.spain", diff = TRUE) +
  # gwesp(decay = 0.5, fixed = TRUE)

#   gwesp(0.25, fixed = T) +
#   gwdsp(0.25, fixed = T)
# gwidegree(cutoff = 90) + 
# gwodegree(cutoff = 70) +
# gwdsp(cutoff = 50) + 
# gwesp(cutoff = 50)

# brunin = 40000
model <- ergm::ergm(formula,
                    control=control.ergm(MCMC.burnin=40000, 
                                         MCMC.interval=10000))
# control=control.ergm(MCMC.burnin=100000, MCMC.samplesize=10000, MCMC.interval=10000)

summary(model)

# Simulate graphs using our ERGM fit
# Compare the number of edges our observed graph has to the average of the simulated networks.

set.seed(1234)
hundred_simulations <- simulate(model, 
                                coef = coef(model),
                                nsim = 100,
                                control = control.simulate.ergm(MCMC.burnin = 1000,
                                                                MCMC.interval = 1000))

net_densities <- unlist(lapply(hundred_simulations, network.density))

hist(net_densities, xlab = "Density", main = "", col = "lightgray")
abline(v = network.density(df_edge_net), col = "red", lwd = 3, lty = 2)
abline(v = mean(net_densities), col = "blue", lwd = 3, lty = 1)



  