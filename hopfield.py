import numpy as np
import random


class Hopfield:
    T = 0
    N = 0
    shapes = []
    network = []
    weights = []

    def __init__(self, N):
        self.N = N
        self.network = np.full((5,10),-1)
    
    def add_shape(self, shape):
        self.shapes.append(shape)
    
    def reset_shapes(self):
        self.shapes = []

    def train_all_shapes(self):
        self.weights = np.zeros((self.shapes[0].size, self.shapes[0].size))
        index = 1
        for net in self.shapes:
            print("training shape%d"%index)
            index += 1 
            net1d = net.flatten()
            self.weights += np.outer(net1d,net1d)

        self.weights /= self.N
        np.fill_diagonal(self.weights,0)
    
    def sync_update(self):
        net1d = self.network.flatten()
        dEnergy = np.matmul(self.weights, self.network.flatten())
        for i in range(self.N*self.N):
            if float(self.T) == 0.0:
                net1d = np.sign(dEnergy)
                i = self.N*self.N
            else:
                rand = random.uniform(0, 1)
                net1d[i] = 1 if rand < 1/(1+np.exp(-dEnergy[i]/self.T)) else -1
        self.network = net1d
        self.network.shape = (self.N,self.N)

    def async_update_ordered(self, iter_step_callback):
        for i in range(self.N*self.N):
            self.update_neuron(i)
            iter_step_callback()

    def async_update_ordered_inv(self, iter_step_callback):
        for i in range(self.N*self.N):
            self.update_neuron(self.N*self.N-1-i)
            iter_step_callback()

    def async_update_shuffeled(self, iter_step_callback):
        list = [x for x in range(self.N*self.N)]
        random.shuffle(list)
        for i in list:
            self.update_neuron(i)
            iter_step_callback()

    def update_neuron(self, i):
        net1d = self.network.flatten()
        dEnergy = np.matmul(self.weights, self.network.flatten())
        if float(self.T) == 0.0:
            net1d[i] = 1 if dEnergy[i]>=0 else -1
        else:
            rand = random.uniform(0, 1)
            net1d[i] = 1 if rand < 1/(1+np.exp(-dEnergy[i]/self.T)) else -1
        self.network = net1d
        self.network.shape = (self.N,self.N)

    def set_temp(self, T):
        self.T = T
    
    def set_network(self, network):
        self.network = network