#!/usr/bin/env python
# -*- coding: utf-8 -*
''''
pesudo code from bible algorithm Actor-critic policy gradient with bootstrapping

'''
from tkinter import N
import gym
from numpy import dtype, ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



class AC(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # conv layer
        # linear layer applies a linear transformation to the incoming data y = xA^t + b
        self.conv_layer = nn.Linear(4,256)
        # dense for pi add a softmax
        # how to represent the dense layer?
        self.pi_layer = nn.Linear(256,2)
        self.v_layer = nn.linear(256,1)
        self.buffer = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self,input,softmax_dim = 0)-> float:

        """
        network pi value function
        :param input: input data for pi network
        :return: 2 probabilitis 
        """
        output1 = F.relu(self.conv_layer(input))
        output2 = self.pi_layer(output1)
        # return probabitliy
        return F.softmax(output2, dim=softmax_dim)

    def v(self,input) -> float:
        output1 = F.relu(self.conv_layer(input))
        output2 = self.v_layer(output1)
        return output2

    def put_into_buffer(self,transition) -> list:
        return self.buffer.append(transition)

    def make_batch(self):
        """
        transfer batch into several lists
        return: batch in tensor datafomat
        """
        s_lst,a_lst,r_lst,s__lst,done_lst = [],[],[],[],[]
        for transition in self.buffer:
            # !!!可以一次性读出来 在读进去！机智啊！！！比tan[0]好
            s,a,r,s_,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/1.00])
            s__lst.append(s_)
            done_lst.append([0.0 if done else 1.0])

        s,a,r,s_,done = torch.tensor(s_lst,dtype=torch.float),\
                torch.tensor(a_lst),torch.tensor(r_lst, dtype=torch.float),\
                torch.tensor(s__lst, dtype=torch.float),\
                torch.tensor(done_lst, dtype=torch.float)

        self.buffer = []
        return s,a,r,s_,done
  

    def train(self):
        # batch 形式
        s, a, r, s_, done = self.make_batch()
        # no gamma on book?
        # td target
        Q = r + self.v(s_)* done
    
        # update network
        # value update
        loss_v = (Q- self.v(s))**2.

        # why take chossen action probability? why all the probability
        # policy update
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss_p = Q * (-torch.log(pi_a))
        
        # why add them together???
        loss = loss_v + loss_p

        loss.mean().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

def main():
    env = gym.make('CartPole-v1')
    ac = AC()

    gamma = 0.98
    learning_rate = 0.0002

    n_depth = 10
    threshould = 0.1
    MAX_EPISODES = 1000
    total_reward = 0.0

    # error and episodes can count as two conditions to jump out of circulation
    for episode in range(MAX_EPISODES):
        state = env.reset()
        done = False

        while not done:

            # start to sample
            for n in range(n_depth):
                
                #how to take actions? following the pi!which is a network
                # the prob data form ??????? what is that?
                prob = ac.pi(torch.from_numpy(state).float())
                # choose action from probability
                # use numpy but something in pytorch have replacement
                #np.random.choice(actions)
                #action = Categorical(prob).sample().item()
                action = Categorical(prob).sample()
                state_,reward, done,info = env.step(action)
                # buffer 不应该 初始化每开一局吗
                # 仔细看 M是episode所以是每一局中 都一直进行这样数据的添加 只不过是 每10次来训练模型一回
                ac.put_data((state,action,reward,state_,done))

                state = state_
                # one episode cumulates its all score
                total_reward += reward

                if done:
                    break
            
            ac.train()

        # if error < threshould:
        #      break

        # every 50 episodes print once
        if episode % 20 == 0 and episode != 0:
            print(f"epidoes:{episode}, average reward in these episodes:{total_reward/20.}")
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()