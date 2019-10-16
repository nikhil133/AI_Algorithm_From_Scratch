"""
Created on Mon Aug 26 20:08:17 2019

@author: Nikhil Nambiar
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
    
def Softmax(x):
    
    j=x-x.max()
    exps=np.exp(j)
   # print(exps/np.sum(exps))
   # print("\n\n",np.sum(exps/np.sum(exps)))
    return exps/np.sum(exps)

def predict(x):
    return Softmax(x)
        


def neuralnet(x,w,t,alpha=0.0001):
   
    x=x.reshape(x.shape[0],1)
    t=t.reshape(t.shape[0],1)
    yhat=np.dot(w,x)
    prob=Softmax(yhat)
    #logit or log odd 
    # probablity of class x when not occured
    # P(X=1)/1-P(X=1)
    error=-(t*np.log(prob))
    
    return prob,error


def gradientDescent(X,Y,w,alpha=0.01):
    run=True
    i=0
    n=X.shape[0]
    cost_list=[]
    cost_list.append(1e10)
    while run:
        
        prob_list=np.zeros([3,1])
        cost=np.zeros([3,1])
    
        for l in range(0,X.shape[0]):
            prob,error=neuralnet(X[l],w,Y[l])
            #appending probablities and error from every neural node of m 
            #samples with n features columnwise
            prob_list=np.append(prob_list,prob,axis=1)
            cost=np.append(cost,error,axis=1)
            
        #removing temporary column from probablity and error list on 
        #every epoch
        prob_list=np.delete(prob_list,0,axis=1)
        
        cost=np.delete(cost,0,axis=1)
        #computing total error rowise i.e total error gained for every class
        tcost=cost.sum(axis=0)
        #computing mean square error cost
        c=(np.dot(tcost.transpose(),tcost)/n)*alpha
        cost_list.append(c)
        
        #back propogation
        prob_list=prob_list.transpose()
        ohat=prob_list-Y
        
        w-=(np.dot(ohat.transpose(),X)/n)*alpha
        
        #terminating training when change in cost is negligible
        if cost_list[i]-cost_list[i+1]<1e-9:
            run=False
        i=i+1
        
    cost_list.pop(0)    
    return cost_list,w    



def main():
    dataset = load_iris()
    X=dataset['data']
    Y=dataset['target']
    Y=Y.reshape(X.shape[0],1)
    np.random.seed(0)
    bias=np.ones(X.shape[0])
    X=np.c_[bias,X]
    yset=dataset['target_names']
    Xt,X_test,Yt,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    onehotencode=OneHotEncoder(categorical_features=[0])
    Yhot=onehotencode.fit_transform(Yt).toarray()
    Ythot=onehotencode.fit_transform(Y_test).toarray()

    plt.scatter(Xt[:,1],Xt[:,2],Xt[:,3],Xt[:,4])
    plt.show()


    w=np.random.rand(Yhot.shape[1],X.shape[1])
    
    classes=dict(enumerate(yset.flatten(), 0))
    costs,w=gradientDescent(Xt,Yhot,w)
    xax=range(len(costs))
    yax=range(len(costs))
    Xax,Yax=np.meshgrid(xax,yax)
    print(w)
   # fig=plt.figure()
    #ax=plt.axes(projection='3d')
    #ax.plot3D(Xax,Yax,costs)
    #plt.show()
    predict_list=[]
    acc=0
    for i in range(0,X_test.shape[0]):
        x=X_test[i]
        x=x.reshape(x.shape[0],1)
        r=np.dot(w,x)
       
        p=predict(r)
        
        j = p.argmax()
        predict_list.append(j)
        k = Ythot[i].argmax()
        print("\n\n")
        if j-k==0:
            acc=acc+1
        print("predicted class ",classes[j])
        print("class which it belongs to",classes[k])     
    
    n=X_test.shape[0]
    ac=(acc/n)*100
    print("accuracy ",ac,"%")

    print(ac)
    print(predict_list)
    plt.scatter(X_test[:,1],X_test[:,2],marker="^",c=predict_list)
    plt.show()
    
    
    
if __name__=="__main__":
    main()


