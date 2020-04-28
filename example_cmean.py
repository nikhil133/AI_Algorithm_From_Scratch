from fuzzy.fuzzy_cmean import clusters
import numpy as np
import math
import matplotlib.pyplot as plt


def main():
    c=2
    g=1.25
    ep=0.01

    dataset=np.array([[0.2,0.4,0.6],[0.4,0.3,0.8],[0.8,0.2,0.5],[0.9,0.5,0.4],
                       [0.6,0.6,0.6],[0.3,0.4,0.5],[0.7,0.6,0.5],[0.2,0.5,0.3],
                       [0.3,0.6,0.8],[0.8,0.3,0.1]],ndmin=2,dtype=float)
    mu=np.array([[0.680551,0.319449],[0.495150,0.504850],[0.821897,0.178103],
                [0.303795,0.696205],[0.333966,0.666034],[0.431538,0.568462],
                [0.415384,0.584616],[0.509643,0.490357],[0.469850,0.530150],
                [0.189164,0.810836]],ndmin=2,dtype=float)
    plt.plot(dataset[:,0],dataset[:,1],'ro')
    plt.show()

    cluster=clusters(dataset,mu,c,g,30)
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


if __name__=='__main__':
    main()