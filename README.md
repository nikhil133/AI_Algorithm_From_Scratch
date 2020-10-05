# AI
AI models 

K-Arm Bandit problem greedy approach

The simplest approach to maximize the reward for an action in RL is the selection of greedy action. That is estimate the value of all possible actions with 

Equation
![alt text](https://github.com/nikhil133/AI_Algorithm_From_Scratch/blob/k-bandit-greedy/formula/bandit_qt0.jpg)

where ![alt text](https://github.com/nikhil133/AI_Algorithm_From_Scratch/blob/k-bandit-greedy/formula/bandit_qt1.jpg)

and select the greedy action with 

Equation ![alt text](https://github.com/nikhil133/AI_Algorithm_From_Scratch/blob/k-bandit-greedy/formula/bandit_qt2.jpg) 

Greedy action selection always exploits current knowledge to maximize immediate reward; it spends no time at all sampling apparently inferior actions to see if they might really be better.
