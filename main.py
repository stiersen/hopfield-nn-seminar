from random import randint
from numpy.lib.histograms import _histogram_dispatcher
from numpy.lib.type_check import imag
import hopfield
from hopfield import Hopfield
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button 
from time import sleep
from rect import SelectionRect
def hopfield60by60():
    image1 = plt.imread("1bitImageRiver.bmp")
    width = image1.shape[0]
    image_ones1 = np.array(list(map(
        lambda x: -1 if x[0]==0 else 1
        , image1.reshape((width*width,4))))).reshape(width,width)
    
    image2 = plt.imread("circle1bit.bmp")
    width = image2.shape[0]
    image_ones2 = np.array(list(map(
        lambda x: -1 if x[0]==0 else 1
        , image2.reshape((width*width,4))))).reshape(width,width)
    h = Hopfield(width)
    h.add_shape(image_ones1)
    h.add_shape(image_ones2)
    h.train_all_shapes()
    h.set_network(image_ones1)
    print("training...")
    update_counter = np.array([0])

    fig, imgax = plt.subplots()
    pltimage = imgax.imshow(h.network)
    # pause_and_update()

    def plt_do_sync_iteration(event):
        hopfield.asci_print(h.network)
        h.sync_update()
        pltimage.set_data(h.network)
        plt.draw()
    def plt_do_ordered_async_iteration(event):
        hopfield.asci_print(h.network)
        h.async_update_ordered(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()
    def plt_do_shuffle_async_iteration(event):
        hopfield.asci_print(h.network)
        h.async_update_shuffeled(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()
    
    def pause_and_update():
        pltimage.set_data(h.network)
        update_counter[0]+=1
        if update_counter[0]%5==0:
            plt.draw()
            plt.pause(0.0000000001)
            update_counter[0]=0

    def add_noise(event=None):
        image_noised = h.network.copy()
        for i in range(200):
            image_noised[randint(0, width-1)][randint(0, width-1)] = randint(0,1)*2-1
        h.set_network(image_noised)
        pltimage.set_data(h.network)
        plt.draw()
    def on_box_release(pos, size):
        image_box_cleared = h.network.copy()
        size = [min(x, h.N) for x in size]
        for x in range(pos[0],pos[0]+size[0]):
            for y in range(pos[1],pos[1]+size[1]):
                image_box_cleared[y][x] = -1
        h.set_network(image_box_cleared)
        pltimage.set_data(h.network)
        plt.draw()
    # plt.draw()
    # plt.imshow(h.network)
    # ax = plt.axes()
    ax = plt.axes([0.2, 0, 0.2, 0.075])
    bsync = Button(ax, 'Do Sync')
    bsync.on_clicked(plt_do_sync_iteration)
    ax = plt.axes([0.4, 0, 0.2, 0.075])
    border = Button(ax, 'Do Asyn Ordered')
    border.on_clicked(plt_do_ordered_async_iteration)
    ax = plt.axes([0.6, 0, 0.2, 0.075])
    bshuffle = Button(ax, 'Do Asyn Shuffle')
    bshuffle.on_clicked(plt_do_shuffle_async_iteration)
    ax = plt.axes([0.8, 0, 0.2, 0.075])
    bnoise = Button(ax, 'Add Noise')
    bnoise.on_clicked(add_noise)


    s_rect = SelectionRect(imgax,fig.canvas,on_box_release)
    s_rect.connect()

    plt.show()

def hopfield4by4():
    h = Hopfield(4)
    cross = np.array([[1,1,-1,1],
    [1,-1,1,-1],
    [1,-1,1,-1],
    [1,1,-1,1]])
    square = np.array([1,1,1,1,
    1,1,-1,1,
    1,1,-1,1,
    1,1,1,1])
    line = np.array([1,-1,-1,-1,
    1,-1,-1,-1,
    1,-1,-1,-1,
    1,-1,-1,-1])
    h.add_shape(square)
    # h.add_shape(line)
    h.add_shape(cross)
    h.train_all_shapes()
    start = np.array([[1,1,1,1],
    [1,1,-1,1],
    [1,1,-1,1],
    [1,-1,-1,-1]])

    h.set_network(start)

    pltimage = plt.imshow(h.network)
    def plt_do_sync_iteration(event):
        hopfield.asci_print(h.network)
        h.sync_update()
        pltimage.set_data(h.network)
        plt.draw()
    def plt_do_ordered_async_iteration(event):
        hopfield.asci_print(h.network)
        h.async_update_ordered(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()
    def plt_do_shuffle_async_iteration(event):
        hopfield.asci_print(h.network)
        h.async_update_shuffeled(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()
        
    def pause_and_update():
        plt.pause(0.02)
        pltimage.set_data(h.network)
        
    # plt.draw()
    # plt.imshow(h.network)
    # ax = plt.axes()
    ax = plt.axes([0.2, 0, 0.2, 0.075])
    bsync = Button(ax, 'Do Sync')
    bsync.on_clicked(plt_do_sync_iteration)
    ax = plt.axes([0.4, 0, 0.2, 0.075])
    border = Button(ax, 'Do Asyn Ordered')
    border.on_clicked(plt_do_ordered_async_iteration)
    ax = plt.axes([0.6, 0, 0.2, 0.075])
    bnext = Button(ax, 'Do Asyn Shuffle')
    bnext.on_clicked(plt_do_shuffle_async_iteration)
    # bprev = Button('Previous')
    # bprev.on_clicked(callback.prev)
    plt.show()


def hopfield3by3():
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
    start = np.array([[1,1,-1],
    [1,-1,1],
    [1,1,-1]])

    h.set_network(start)

    pltimage = plt.imshow(h.network)
    def plt_reset(event):
        start = np.array([[1,1,-1],
        [1,-1,1],
        [1,1,-1]])
        h.set_network(start)
        pltimage.set_data(h.network)
        plt.draw()
    def plt_do_sync_iteration(event):
        hopfield.asci_print(h.network)
        h.sync_update()
        pltimage.set_data(h.network)
        plt.draw()
    def plt_do_ordered_async_iteration(event):
        hopfield.asci_print(h.network)
        h.async_update_ordered(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()
    def plt_do_inv_ordered_async_iteration(event):
        hopfield.asci_print(h.network)
        h.async_update_ordered_inv(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()
    def plt_do_shuffle_async_iteration(event):
        hopfield.asci_print(h.network)
        h.async_update_shuffeled(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()
        
    def pause_and_update():
        plt.pause(0.02)
        pltimage.set_data(h.network)
        
    # plt.draw()
    # plt.imshow(h.network)
    # ax = plt.axes()
    ax = plt.axes([0.0, 0, 0.2, 0.075])
    bsync = Button(ax, 'Do Sync')
    bsync.on_clicked(plt_do_sync_iteration)
    ax = plt.axes([0.2, 0, 0.2, 0.075])
    b_order = Button(ax, 'Ordered')
    b_order.on_clicked(plt_do_ordered_async_iteration)
    ax = plt.axes([0.4, 0, 0.2, 0.075])
    b_orderinv = Button(ax, 'Inv Ordered')
    b_orderinv.on_clicked(plt_do_inv_ordered_async_iteration)
    ax = plt.axes([0.6, 0, 0.2, 0.075])
    bnext = Button(ax, 'Do Asyn Shuffle')
    bnext.on_clicked(plt_do_shuffle_async_iteration)
    ax = plt.axes([0.8, 0, 0.2, 0.075])
    breset = Button(ax, 'Reset')
    breset.on_clicked(plt_reset)
    # bprev = Button('Previous')
    # bprev.on_clicked(callback.prev)
    plt.show()



# hopfield4by4()
# hopfield3by3()
hopfield60by60()
