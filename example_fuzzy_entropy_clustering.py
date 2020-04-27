from fuzzy.entropyclustering import cal_fuzzy_entropy,clustering
import numpy as np
import math
import matplotlib.pyplot as plt

dataset=np.array([[0.2,0.4,0.6],[0.4,0.3,0.8],[0.8,0.2,0.5],[0.9,0.5,0.4],[0.6,0.6,0.6],[0.3,0.4,0.5],
[0.7,0.6,0.5],[0.2,0.5,0.3],[0.3,0.6,0.8],[0.8,0.3,0.1]],ndmin=2,dtype=float)
beta=0.5
plt.plot(dataset[:,0],dataset[:,1],'ro')
plt.show()

E,Resrc=cal_fuzzy_entropy(dataset)
cluster=clustering(Resrc,E,beta)

cx=[]
cy=[]
for cpt in cluster:
    x=[]
    y=[]    
    for c in cluster[cpt]:
        x.append(dataset[int(c)][0])
        y.append(dataset[int(c)][1])
    cx.append(x)
    cy.append(y)        

for i in range(0,len(cx)):
    plt.scatter(cx[i],cy[i],cmap=[i,i])
plt.show()