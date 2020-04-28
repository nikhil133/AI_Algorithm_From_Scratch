import numpy as np 
import math

def clustercenter(dataset,mu,g,c,l):
    CC=[]
    C=[]
    for j in range(0,mu.shape[1]):
        for i in range(0,dataset.shape[1]):
            CC.append((sum(mu[:,j]**g*dataset[:,i]))/(sum(mu[:,j]**g)))

    CC=np.array(CC)        
    C=CC.reshape(c,l)
    return C

def dist_data_cc(dataset,C,c):
    d=[]
    for i in range(0,dataset.shape[0]):
        for j in range(0,C.shape[0]):
            d.append(math.sqrt(sum((C[j]-dataset[i])**2)))
    d=np.array(d)
    d=d.reshape(dataset.shape[0],c)
    return d    

def update_mu(D,mu,g):
    r=2/(g-1)
    
    for i in range(0,D.shape[0]):
        for j in range(0,D.shape[1]):
            sum=0.0
            for m in range(0,mu.shape[1]):
               sum+=(D[i,j]/D[i,m])**r
            cal_mu=1/sum
            mu[i,j]=cal_mu
            
    return mu

def modified_membership(dataset,mu,c,g,epoch):
    l=dataset.shape[1]    
    ittr=0
    prev_mu=mu

    while True:
        C=clustercenter(dataset,mu,g,c,l)
        D=dist_data_cc(dataset,C,c)
        mu=update_mu(D,mu,g)
        if epoch<=0:
            if (prev_mu-mu).all()<ep:
                break
        elif epoch==ittr:
            break
        
        prev_mu=mu
        ittr+=1
    return mu

#print("itteration ",ittr)
def clusters(dataset,mu,c,g,epoch):
    mu=modified_membership(dataset,mu,c,g,epoch)
    cluster_pos=np.argmax(mu,axis=1)
    c=set(cluster_pos)
    cluster=dict()
    for i,data in enumerate(cluster_pos):
        if data in cluster.keys():	
            cluster[data].append(i)
        else:
            cluster[data]=[i]
    return cluster


