"""
Simple test to illustrate how the target function could be used.
Here focussing on the 2ND MOMENT of the no. of infected animals,
and using a Waxman network
"""

import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

print(f.__doc__)

# set some seed to get reproducible results:
set_seed(1)

print("Preparing the target function for a Waxman-graph-based, fixed transmissions network")

# generate a Waxman graph:
waxman = nx.waxman_graph(120)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

# at the beginning, call prepare() once:
f.prepare(
    static_network=static_network,  
    capacity_distribution=np.random.lognormal,  # this is more realistic than a uniform distribution
    delta_t_symptoms=60
    )
n_inputs = f.get_n_inputs()
print("n_inputs (=number of network nodes):", n_inputs)
total_budget = n_inputs


evaluation_parms = { 
        'n_simulations': 1000, 
        'statistic': lambda a: np.mean(a**2) #lambda a: np.percentile(a, 95)
        }

"""
Simple test to illustrate how the target function could be used.
Here focussing on the 2ND MOMENT of the no. of infected animals,
and using a Waxman network
"""

# set some seed to get reproducible results:
set_seed(1)

# generate a Waxman graph:
waxman = nx.waxman_graph(120)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
G = nx.DiGraph(nx.to_numpy_array(waxman))
n_trials = 100

degree_values = sorted(G.degree, key=lambda x: x[1], reverse=True)

#for s in G.degree():
    #print(s)
ob = [1.0151099272561344,1.0391939579942095,0.9876526704687878,0.9532095062846853,1.0571130433956615,0.9152694175850806,1.1336670304679093,0.8726170673485641,0.9484660107247245,0.8844693921580916,0.9303699602501242,0.9886871684271935,0.9748819242921599,0.8475260465191491,1.018616284482729,1.0503080080148854,0.942477395743531,0.9564271267781319,1.0926369027462015,0.9511871472843774,1.0467949532415637,0.967019923926813,0.9984550322381032,0.9676919018344006,1.0705742745507847,1.0216417121004153,1.0030780883363357,1.0971455496574747,1.209051759292525,1.5614094210261171,0.9448527677299784,1.1145168892760968,1.0016807711783309,0.9353394849704806,1.219687073568178,0.9390751128724375,1.0131004530477814,0.9769588625476427,1.0784511105658856,0.9571531849858572,0.9786090981976783,0.8626875690888361,1.1005933864160578,0.8918922267628813,0.8800107756731245,0.9185112260727595,0.9640364983777284,0.9791083050634549,1.0308127163245617,0.9964726940880929,0.8672943400804966,1.076102652702318,0.8950409528316093,
1.0357119423982546,1.0119542661728,1.1469596302863532,0.9028412894340175,1.2622068929540848,1.0106059958001208,0.9793246227380834,0.9673434102641777,0.9511802116624637,
0.878958828161181,0.876642275830323,1.1198355035553504,0.9578499859673222,0.8800853673582595,0.9917622365886624,0.9664390781812204,0.968062883335317,0.9529102498823236,
1.0459597477921785,1.010812086871432,0.8797531770959931,0.9584216956681293,1.1167993688272844,1.2402799471691077,0.898087986356128,0.9764460209706127,0.8825330029160444,
0.8691612521895462,1.0418070598202753,0.9583669399494759,1.070453505295795,1.1364539677562062,0.969545880575111,1.07559634205368,0.8993172931393597,1.0108931291667715,
0.8568592747602155,0.9376178557114399,1.064579599491093,1.1273630896376305,1.1697265964191865,1.0225991527741047,1.0401814686956585,0.9843633964436199,0.9715002477358362,
1.137379685892998,1.0874900529093614,1.0448511584716853,1.0988816936700792,1.1020795171498592,0.981020182343382,1.0660180742844931,0.9230220295313853,0.9513222680706985,
0.9341573422745043,0.9588306849538698,1.0803066698554902,0.8744426417833885,0.9595296356266911,0.8973968785293814,0.9795144053160978,0.917352362718618,1.0461504788232978,
0.9229428286271538,0.9074525009359177,0.918153902951505,1.0108144925787819]

hd = []
for i in range(10):
  hd.append(degree_values[i][0])

#83, 12, 27, 31, 96, 113, 30, 14,49 

def stderr(a):
    return np.std(a, ddof=1) / np.sqrt(np.size(a))


sentinels = hd
#sentinels = [33, 36, 63, 66]
weights = np.zeros(n_inputs)
weights[sentinels] = 1
shares = weights / weights.sum()

x4 = shares * total_budget
x4max = x4.max()
plt.figure()
#plt.show()

y1 = np.array([f.evaluate(x4, **evaluation_parms) for it in range(n_trials)])
z1 = np.array([f.evaluate(ob, **evaluation_parms) for it in range(n_trials)])
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(y1.mean()), stderr(y1))
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(z1.mean()), stderr(z1))


y2 = np.array([f.evaluate(x4, **evaluation_parms) for it in range(n_trials)])
z2 = np.array([f.evaluate(ob, **evaluation_parms) for it in range(n_trials)])
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(y2.mean()), stderr(y2))
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(z2.mean()), stderr(z2))


y3 = np.array([f.evaluate(x4, **evaluation_parms) for it in range(n_trials)])
z3 = np.array([f.evaluate(ob, **evaluation_parms) for it in range(n_trials)])
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(y3.mean()), stderr(y3))
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(z3.mean()), stderr(z3))


y4 = np.array([f.evaluate(x4, **evaluation_parms) for it in range(n_trials)])
z4 = np.array([f.evaluate(ob, **evaluation_parms) for it in range(n_trials)])
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(y4.mean()), stderr(y4))
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(z4.mean()), stderr(z4))


y5 = np.array([f.evaluate(x4, **evaluation_parms) for it in range(n_trials)])
z5 = np.array([f.evaluate(ob, **evaluation_parms) for it in range(n_trials)])
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(y5.mean()), stderr(y5))
print("\nMean and std.err. of", n_trials, "evaluations at a sentinel-based x:", np.sqrt(z5.mean()), stderr(z5))
