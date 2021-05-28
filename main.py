import hopfield
from hopfield import Hopfield
import numpy as np

h = Hopfield(3)
cross = np.array([[1,-1,1],
[-1,1,-1],
[1,-1,1]])
square = np.array([1,1,1,
1,-1,1,
1,1,1])
h.add_shape(square)
h.add_shape(cross)
h.train_all_shapes()
start = np.array([[-1,1,-1],
[1,-1,1],
[-1,1,-1]])
h.set_network(start)
while True:
    hopfield.asci_print(h.network)
    h.update()
    input()