### PARAMETERS:

folder = "/home/heitzig/data/optes/2022_05/"
    
transmissions_file = folder + "dataset.txt"
capacity_file = folder + "barn_size.txt"
n_breeding = 500
n_fattening = 400
n_trader = 55
n_slaughterhouse = 85


case = "GP"

if case == "MCMC":
    budget_type = "sentinels"
    sentinels_file = folder + "mcmc_sentil.txt"
    total_budget = 1040
elif case == "GP":
    budget_type = "npy"
    budget_file = folder + "gpgo_95perc_nodes_1040.npy"
elif case == "DRL":
    budget_type = "csv"
    budget_file = folder + "DRL_optimal_budget_1040_2.csv"
    
###############


import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
import networkx as nx
from pyunicorn import Network
from scipy import stats
import statsmodels.api as sm

from pyoptes.networks.transmissions import extract_static, TransmissionEvent, Transmissions

# network:
    
D = np.loadtxt(transmissions_file, delimiter=",", dtype="int")
sent = received = D[:, 2]
source = D[:, 0]
target = D[:, 1]
size = D[:, 3]
n_transmissions = len(sent)
T = Transmissions(180, [
        TransmissionEvent(sent[i], received[i], source[i], target[i], size[i])
        for i in range(n_transmissions)
    ])
G = extract_static(T)
print("number of transmissions", n_transmissions)
print("number of edges", G.number_of_edges())

# capacities:
    
with open(capacity_file) as f:
    capacity = np.array(f.readline().split(",")[:-1], "float")
N = capacity.size
print("number of nodes", N)
print("total capacity", capacity.sum())
print("average capacity", capacity.mean())

# type dummies:

is_breeding = np.zeros(N)
is_breeding[:n_breeding] = 1
is_fattening = np.zeros(N)
is_fattening[n_breeding:n_breeding+n_fattening] = 1
is_trader = np.zeros(N)
is_trader[n_breeding+n_fattening:n_breeding+n_fattening+n_trader] = 1

# budget:
    
if budget_type == "sentinels":
    with open(sentinels_file) as f:
        sentinels = np.array(f.readline().split(",")[:-1], "int")
    budget = np.zeros(N)
    budget[sentinels] = total_budget / len(sentinels)
elif budget_type == "npy":
    budget = np.load(budget_file)
    total_budget = budget.sum()
elif budget_type == "csv":
    budget = np.loadtxt(budget_file, delimiter=",")[:,1]
    total_budget = budget.sum()
else: 
    with open(budget_file) as f:
        budget = np.array(f.readline().split(",")[:-1])
    total_budget = budget.sum()
gets_large_budget = 1 * (budget > budget.mean())
print("total budget", budget.sum())
print("farms getting a large budget", gets_large_budget.sum())

# network metrics:
    
degree = np.zeros(N)
in_degree = np.zeros(N)
out_degree = np.zeros(N)
in_transmissions = np.zeros(N)
out_transmissions = np.zeros(N)
in_ev_centrality = np.zeros(N)
out_ev_centrality = np.zeros(N)
in_closeness = np.zeros(N)
out_closeness = np.zeros(N)
sp_betweenness = np.zeros(N)
adj = np.zeros((N,N))
for (v, x) in G.degree: degree[v] = x
for (v, x) in G.in_degree: in_degree[v] = x
for (v, x) in G.out_degree: out_degree[v] = x
for i in range(n_transmissions): 
    in_transmissions[target[i]] += 1
    out_transmissions[source[i]] += 1
    adj[source[i],target[i]] = 1
for (v, x) in nx.eigenvector_centrality_numpy(G).items(): in_ev_centrality[v] = x
for (v, x) in nx.eigenvector_centrality_numpy(G.reverse()).items(): out_ev_centrality[v] = x
for (v, x) in nx.closeness_centrality(G).items(): in_closeness[v] = x
for (v, x) in nx.closeness_centrality(G.reverse()).items(): out_closeness[v] = x
for (v, x) in nx.betweenness_centrality(G).items(): sp_betweenness[v] = x

net = Network(adj, directed=True, node_weights=capacity)
net_rev = Network(adj.T, directed=True, node_weights=capacity)
nsi_degree = net.nsi_degree()
nsi_in_degree = net.nsi_indegree()
nsi_out_degree = net.nsi_outdegree()
#nsi_in_closeness = net_rev.nsi_closeness()
#nsi_out_closeness = net.nsi_closeness()
#nsi_sp_betweenness = net.nsi_betweenness()
nsi_rw_betweenness = net.nsi_newman_betweenness()


# data table, ignoring slaughterhouses:

df = pd.DataFrame({
    "budget": budget,
    "gets large budget": gets_large_budget,
    "capacity": capacity,
    "is breeding": is_breeding,
    "is fattening": is_fattening,
    "is trader": is_trader,
    "partners": degree,
    "suppliers": in_degree,
    "customers": out_degree,
    "transmissions received": in_transmissions,
    "transmissions sent": out_transmissions,
    "in-eigenvector centrality": in_ev_centrality,
    "out-eigenvector centrality": out_ev_centrality,
    "in-closeness": in_closeness,
    "out-closeness": out_closeness,
    "shortest-path betweenness": sp_betweenness,
    "n.s.i. degree": nsi_degree,
    "n.s.i. in-degree": nsi_in_degree,
    "n.s.i. out-degree": nsi_out_degree,
#    "n.s.i. in-closeness": nsi_in_closeness,
#    "n.s.i. out-closeness": nsi_out_closeness,
#    "n.s.i. shortest-path betweenness": nsi_sp_betweenness,
    "n.s.i. random-walk betweenness": nsi_rw_betweenness,
}).iloc[:-n_slaughterhouse,:]

print(df)

df.boxplot()
plt.show()

# standardization:

df_z = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)

indeps = [
    "capacity", 
    "suppliers", "customers", "transmissions received", "transmissions sent", 
    "in-eigenvector centrality", "out-eigenvector centrality", 
    "in-closeness", "out-closeness",
    "shortest-path betweenness",
    "n.s.i. degree", "n.s.i. in-degree", "n.s.i. out-degree",
#    "n.s.i. in-closeness", "n.s.i. out-closeness",
#    "n.s.i. shortest-path betweenness",
    "n.s.i. random-walk betweenness",
]
dummies = ["is breeding", "is fattening"]

# regression for budget without type:
    
Y = df_z["budget"]
X = df_z[indeps]
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

# regression for budget with type:
    
Y = df_z["budget"]
X = df_z[indeps + dummies]
X = sm.add_constant(X) 

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

# regression for gets_large_budget on capacity and partners alone:
    
Y = df["gets large budget"] 
X = df_z[["capacity", "partners"]] 
X = sm.add_constant(X) 

model = sm.Logit(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)


# regression for gets_large_budget without type:
    
Y = df["gets large budget"] 
X = df_z[indeps]
X = sm.add_constant(X) 

model = sm.Logit(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

# regression for gets_large_budget on capacity and sp betweenness only:
    
Y = df["gets large budget"] 
X = df_z[["capacity", "shortest-path betweenness"]]
X = sm.add_constant(X) 

model = sm.Logit(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)


# regression for gets_large_budget on capacity and n.s.i. degree only:
    
Y = df["gets large budget"] 
X = df_z[["capacity", "n.s.i. degree"]]
X = sm.add_constant(X) 

model = sm.Logit(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

# regression for gets_large_budget on capacity and n.s.i. rw-betw. only:
    
Y = df["gets large budget"] 
X = df_z[["capacity", "n.s.i. random-walk betweenness"]]
X = sm.add_constant(X) 

model = sm.Logit(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)


# plots:
    
#sns.heatmap(df.corr(), annot=True, fmt=".2f")
#sns.scatterplot(budget, degree)
#plt.show()


