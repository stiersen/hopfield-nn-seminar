#!/usr/bin/env python3
import numpy as np
import random

from setuptools.command.bdist_egg import iter_symbols

"""
adds the network at the desired position to the asci canvas
"""
def asci_printer_on_canvas(position_x, position_y, np_network, np_asci_canvas):
    if np_network.shape[0] + position_x > np_asci_canvas.shape[0]:
        print("ERROR: asci canvas to small width")
    if np_network.shape[1] + position_y > np_asci_canvas.shape[1]:
        print("ERROR: asci canvas to small height")
    for x in range(np_network.shape[0]):
        for y in range(np_network.shape[1]):
            np_asci_canvas[x+position_x,y+position_y] = np_network[x,y]

def asci_print(np_asci_canvas):
    str = ""
    shape = np_asci_canvas.shape
    if len(shape) == 1:
        for x in range(np_asci_canvas.size):
            str+= " " if np_asci_canvas[x] < 0 else "#"
    else:
        for x in range(np_asci_canvas.shape[0]):
            for y in range(np_asci_canvas.shape[1]):
                str+= " " if np_asci_canvas[x,y] <= 0 else "#"
            str += "\n"
    print(str)

def get_as_1D(np_arr):
    return np.reshape(np_arr, (1,np_arr.size))
def get_as_2D(np_arr, width):
    return np.reshape(np_arr.copy(), (width,int(np_arr.size/width)))

# def train_weights(sample_list):
#     weights = np.zeros((sample_list[0].size,sample_list[0].size))
#     for network in sample_list:
#         net1d = get_as_1D(network)
#         for i in range(net1d.size):
#             print("row %d"% i)
#             for k in range(net1d.size):
#                 weights[i, k] += net1d[0,i] * net1d[0,k]
#     np.fill_diagonal(weights,0)
#     return weights

def calc_test():
    cross = np.array([[1,-1,1],
    [-1,1,-1],
    [1,-1,1]])
    square = np.array([1,1,1,
    1,-1,1,
    1,1,1])

    sample_list = [cross,square]
    print(sample_list)
    cross = np.array(cross)
    # asci_print(cross)
    square_shaped = np.array(get_as_2D(np.array(square), 3))
    # asci_print(np.array(square))
    # asci_print(square_shaped)

    canvas = np.full((5,10),-1)
    asci_printer_on_canvas(1,0,square_shaped,canvas)
    asci_printer_on_canvas(1,5,cross,canvas)
    weights = train_weights(sample_list)
    asci_print(weights)
    # asci_print(canvas)

class Hopfield:
    N = 0
    shapes = []
    network = []
    weights = []
    def __init__(self, N):
        self.N = N
        self.network = np.full((5,10),-1)
    
    def add_shape(self, shape):
        self.shapes.append(shape)

    def train_all_shapes(self):
        self.weights = np.zeros((self.shapes[0].size, self.shapes[0].size))
        index = 1
        for net in self.shapes:
            print("training shape%d"%index)
            index += 1 
            net1d = get_as_1D(net)
            self.weights += np.outer(net1d[0],net1d[0])
        np.fill_diagonal(self.weights,0)
    
    def set_network(self, network):
        self.network = network

    def sync_update(self):
        net1d = get_as_1D(self.network)
        new_net = np.matmul(self.weights, self.network.flatten()) #W S from eq 1 hopfield.pdf

        # 1 1 1 1     1
        # 1 1 1 1     2
        # 1 1 1 1     3
        # 1 1 1 1     2
        # wird das neuron nicht geaendert, wenn die Summe der weights = 0 ist
        li = list(map(
            lambda el: net1d[0][el[0]] if el[1]==0 else (-1 if el[1] < 0 else 1)
            , enumerate(new_net)))
        # li_alt = list(map( lambda x: -1 if x < 0 else 1,new_net))
        self.network = np.array(li)
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