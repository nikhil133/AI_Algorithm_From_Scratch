import numpy as np
import math

def eculidean_dist(dataset):
    dist=[]
    resd_dist=[]
    index=[]
    resd_index=[]
    for i,datai in enumerate(dataset):
        for j,dataj in enumerate(dataset):
            if j!=i and i<j:
                s=math.sqrt(sum((datai-dataj)**2))
                dist.append(s)
                index.append([i,j])
            elif j!=i and i>j:
                rs=math.sqrt(sum((datai-dataj)**2))
                resd_dist.append(rs)
                resd_index.append([i,j])
    
    total_dist=sum(dist)
    dist=np.array(dist)
    index=np.array(index)
    resd_dist=np.array(resd_dist)
    resd_index=np.array(resd_index)
    n_dist=dist.shape[0]
    mean_d=total_dist/n_dist
    return dist,index,mean_d,resd_dist,resd_index

def similarity(dist,mean_d):
    alpha=math.log(2)/mean_d
    sim=[math.exp(-alpha*d)for d in dist]
    return np.array(sim)

def cal_entropy(resources,rresources):
    resource=np.vstack((resources,rresources))
    res=dict()
    for r in resource:
        if r[0] in res.keys():
            res[r[0]]+=-r[3]*math.log2(r[3])-(1-r[3])*math.log2(1-r[3])
        else:
            res[r[0]]=-r[3]*math.log2(r[3])-(1-r[3])*math.log2(1-r[3])
    return res


def cal_fuzzy_entropy(dataset):
    dist,index,dmean,rdist,rindex=eculidean_dist(dataset)
    resources=np.c_[index,dist]
    rresources=np.c_[rindex,rdist]
    sim=similarity(dist,dmean)
    rsim=similarity(rdist,dmean)
    resources=np.c_[resources,sim]
    rresources=np.c_[rresources,rsim]
    entropy=cal_entropy(resources,rresources)
    return entropy,resources


def clustering(data,entropy,threshold):
    cluster=dict()

    m=min(entropy,key=entropy.get)
    cdata=[]
    cdata.append(m)
    i=0
    while True:
        if int(m)==data[i][0]:
            if threshold<=data[i][3]:
                cdata.append(data[i][1])
                cluster[m]=cdata
        else:
            del entropy[m]
            if bool(entropy):
                m=min(entropy,key=entropy.get)
            else:
                break

            cdata=[]
            cdata.append(m)
            if int(m)==data[i][0]:
                if threshold<=data[i][3]:
                    cdata.append(data[i][1])
                    cluster[m]=cdata
        if i<len(data):
            i+=1
        else:
            i=0
    return cluster

