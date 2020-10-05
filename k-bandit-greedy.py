import numpy as np
import matplotlib.pyplot as plt
actions_data=np.array([1,2,3,4,5,3,3,2,1,4,2])
actions=set(actions_data)
rewards=np.array([0,0,1,1,0,1,1,0,1,0,0])
timestamp=100
t=1
avg_action_value=[]
def isAction(At,a):
    if At==a:
        return 1
    return 0


while t!=timestamp-1:
    action_est_t=np.array([0]*5)
    action_performed_t=np.array([0]*5)
        
    for action in actions:
        for index,action_list in enumerate(actions_data):
            action_est_t[action-1]+=rewards[index]*isAction(action,action_list)
            action_performed_t[action-1]+=isAction(action,action_list)
    avg_action_value.append(action_est_t/action_performed_t)    
    t+=1

total_avg_action_value=np.sum(avg_action_value,axis=0)
print(total_avg_action_value)
At=np.argmax(total_avg_action_value)
bins=np.array(list(actions))
plt.xlabel("arm pulled")
plt.ylabel("reward")
plt.bar(bins,total_avg_action_value)
plt.show()
