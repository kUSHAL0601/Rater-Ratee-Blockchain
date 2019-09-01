import networkx as nx
import math

def initiliaze_scores(G):
    fairness = {}
    goodness = {}
    
    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight')*1.0/G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness

def compute_fairness_goodness(G):
    fairness, goodness = initiliaze_scores(G)
    
    nodes = G.nodes()
    iter = 0
    while iter < 100:
        df = 0
        dg = 0

        print('-----------------')
        print("Iteration number", iter)
        
        print('Updating goodness')
        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]]*edge[2]
            try:
                dg += abs(g/len(inedges) - goodness[node])
                goodness[node] = g/len(inedges)
            except:
                pass

        print('Updating fairness')
        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]])/2.0
            try:
                df += abs(f/len(outedges) - fairness[node])
                fairness[node] = f/len(outedges)
            except:
                pass
        
        print('Differences in fairness score and goodness score = %.2f, %.2f' % (df, dg))
        if df < math.pow(10, -6) and dg < math.pow(10, -6):
            break
        iter+=1
    
    return fairness, goodness

G = nx.DiGraph()

f = open("./db/BTCAlphaNet.csv","r")
avg_rater={}
avg_ratee={}
for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2])) ## the weight should already be in the range of -1 to 1
    try:
        avg_rater[ls[0]].append(float(ls[2]))
    except:
        avg_rater[ls[0]]=[float(ls[2])]
    try:
        avg_ratee[ls[1]].append(float(ls[2]))
    except:
        avg_ratee[ls[1]]=[float(ls[2])]
f.close()

fairness, goodness = compute_fairness_goodness(G)

# Now we predict the malicious rater's and ratee's assuming the ground truth that > 0.5 rating denotes a 

for i in avg_rater:
    avg_rater[i]=sum(avg_rater[i])/len(avg_rater[i])

for i in avg_ratee:
    avg_ratee[i]=sum(avg_ratee[i])/len(avg_ratee[i])

import numpy as np
rater=[]
label_rater=[]
train_set_rater=[]
for i in avg_rater:
    rater.append(i)
    if avg_rater[i]>=0.5:
        label_rater.append(1)
    else:
        label_rater.append(-1)
    train_set_rater.append([fairness[i],goodness[i]])

train_set_rater=np.array(train_set_rater)
label_rater=np.array(label_rater)


ratee=[]
label_ratee=[]
train_set_ratee=[]
for i in avg_ratee:
    ratee.append(i)
    if avg_ratee[i]>=0.5:
        label_ratee.append(1)
    else:
        label_ratee.append(-1)
    train_set_ratee.append([fairness[i],goodness[i]])

train_set_ratee=np.array(train_set_ratee)
label_ratee=np.array(label_ratee)


from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(train_set_rater, label_rater, test_size = 0.2, random_state = 0)

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

clf1=GaussianProcessClassifier(1.0 * RBF(1.0))
clf1.fit(xTrain, yTrain)
score = clf1.score(xTest, yTest)
print("Rater",score)

xTrain, xTest, yTrain, yTest = train_test_split(train_set_ratee, label_ratee, test_size = 0.2, random_state = 0)

clf1=GaussianProcessClassifier(1.0 * RBF(1.0))
clf1.fit(xTrain, yTrain)
score = clf1.score(xTest, yTest)
print("Ratee",score)
