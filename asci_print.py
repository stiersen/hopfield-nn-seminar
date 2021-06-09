
import numpy as np
import random
from random import randint
import array

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
