import hopfield
from hopfield import Hopfield
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button 
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
[1,1,-1]])

h.set_network(start)

def plt_do_iteration(event):
    hopfield.asci_print(h.network)
    h.update()
    plt.imshow(h.network)
    plt.draw()

plt.imshow(h.network)
axprev = plt.axes()
# axprev = plt.axes([0.5, 0, 0.1, 0.075])
bnext = Button(axprev, 'Do Iteration')
bnext.on_clicked(plt_do_iteration)
# bprev = Button('Previous')
# bprev.on_clicked(callback.prev)

plt.show()