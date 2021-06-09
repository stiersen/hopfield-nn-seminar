
import numpy as np
import random
from random import randint


def get_as_1D(np_arr):
    return np.reshape(np_arr, (1,np_arr.size))
def get_as_2D(np_arr, width):
    return np.reshape(np_arr.copy(), (width,int(np_arr.size/width)))


class Hopfield:
    N = 0
    shapes = []
    network = []
    weights = []
    weight_noncorrupted = []
    total_brain_damage = 0
    def __init__(self, N):
        self.N = N
        self.network = np.full((5,10),-1)
    
    def add_shape(self, shape):
        self.shapes.append(shape)

    def set_brain_damage(self, percent):
        demaged_weights = self.weights.flatten()
        count = self.N**4
        choices = [0 if x<percent/10 else 1 for x in range(10)]
        picker = np.random.choice(choices, 2560000)
        self.weights = (demaged_weights * picker).reshape(self.N**2, self.N**2)

    def reset_damage_brain(self):
        self.weights = self.weight_noncorrupted
        self.total_brain_damage = 0
    
    def train_all_shapes(self):
        self.weights = np.zeros((self.shapes[0].size, self.shapes[0].size))
        index = 1
        for net in self.shapes:
            print("training shape%d"%index)
            index += 1 
            net1d = get_as_1D(net)

            #-------------------sum up all outer products------
            self.weights += np.outer(net1d[0],net1d[0])
            #--------------------------------------------------

        np.fill_diagonal(self.weights,0)
        self.weight_noncorrupted = self.weights.copy()
    

    def sync_update(self):
        net1d = get_as_1D(self.network)

        #-----------update with matrix multiplication---------
        new_net = np.matmul(self.weights, self.network.flatten()) #W S from eq 1 hopfield.pdf
        #------------------heavyside/sign func----------------
        self.network = np.sign(new_net)
        #-----------------------------------------------------
        
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
        net1d = get_as_1D(self.network)
        new_net = np.matmul(self.weights, self.network.flatten()) #W S from eq 1 hopfield.pdf
        net1d[0][i] = net1d[0][i] if new_net[i]==0 else (-1 if new_net[i] < 0 else 1)
        # net1d[0][i] = -1 if new_net[i] < 0 else 1
        self.network = net1d
        self.network.shape = (self.N,self.N)
    
    def set_network(self, network):
        self.network = network

