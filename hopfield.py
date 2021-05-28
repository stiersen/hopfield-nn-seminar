#!/usr/bin/env python3
import numpy as np
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
    return np.reshape(np_arr.copy(), (1,np_arr.size))
def get_as_2D(np_arr, width):
    return np.reshape(np_arr.copy(), (width,int(np_arr.size/width)))

def train_weights(sample_list):
    weights = np.zeros((sample_list[0].size,sample_list[0].size))
    for network in sample_list:
        net1d = get_as_1D(network)
        for i in range(net1d.size):
            for k in range(net1d.size):
                weights[i, k] += net1d[0,i] * net1d[0,k]
    np.fill_diagonal(weights,0)
    return weights

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
            for i in range(net1d.size):
                for k in range(net1d.size):
                    w_ij = net1d[0,i] * net1d[0,k]
                    self.weights[i, k] += w_ij
                    self.weights[k, i] += w_ij # It would be possible to also ignore the lower half triangle entirely.
        np.fill_diagonal(self.weights,0)
    
    def set_network(self, network):
        self.network = network

    def update(self):
        # net1d = get_as_1D(self.network)
        new_net = np.matmul(self.weights, self.network.flatten()) #W S from eq 1 hopfield.pdf
        li = list(map(lambda el: net1d[el[0]] if el[1]==0 else (-1 if el[1] < 0 else 1), enumerate(new_net)))
        self.network = np.array(li)
        self.network.shape = (self.N,self.N)