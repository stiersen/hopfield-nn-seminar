from random import randint
from hopfield import Hopfield
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from rect import SelectionRect


def asterixObelix():
    image_paths = ["asterixObelixChars/asterix.bmp",
        "asterixObelixChars/idefix.bmp",
        "asterixObelixChars/miraculix.bmp",
        "asterixObelixChars/obelix.bmp"]
    N = plt.imread(image_paths[0]).shape[0]
    h = Hopfield(N)

    for path in image_paths:
        image = plt.imread(path)
        image_ones = np.array(list(map(
            lambda x: 1 if x[0]==0 else -1
            , image.reshape((N*N,4))))).reshape(N,N)
        h.add_shape(image_ones)

    h.train_all_shapes()
    h.set_network(h.shapes[0])

    fig, imgax = plt.subplots()
    imgax.set_position([0.2,0.16,0.8,0.8])
    pltimage = imgax.imshow(h.network, cmap=plt.get_cmap("binary"))

    def plt_do_sync_iteration(event):
        h.sync_update()
        pltimage.set_data(h.network)
        plt.draw()

    def plt_do_ordered_async_iteration(event):
        h.async_update_ordered(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()

    def plt_do_shuffle_async_iteration(event):
        h.async_update_shuffeled(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()

    def change_temp(event):
        h.set_temp(s_temp.val)
    
    update_counter = np.array([0])
    def pause_and_update():
        pltimage.set_data(h.network)
        update_counter[0]+=1
        if update_counter[0]%80==0:
            plt.draw()
            plt.pause(0.0000000001)
            update_counter[0]=0

    def add_noise(event=None):
        image_noised = h.network.copy()
        for i in range(200):
            image_noised[randint(0, N-1)][randint(0, N-1)] = randint(0,1)*2-1
        h.set_network(image_noised)
        pltimage.set_data(h.network)
        plt.draw()

    def on_box_release(pos, size):
        image_box_cleared = h.network.copy()
        size = [min(x, h.N) for x in size]
        for x in range(pos[0],pos[0]+size[0]):
            for y in range(pos[1],pos[1]+size[1]):
                image_box_cleared[y][x] = 1
        h.set_network(image_box_cleared)
        pltimage.set_data(h.network)
        plt.draw()

    def random_memory(event):
        h.network = h.shapes[randint(0,len(h.shapes)-1)]
        pltimage.set_data(h.network)
        plt.draw()

    def invert_network(event):
        h.network = h.network*-1
        pltimage.set_data(h.network)
        plt.draw()
    
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
    ax = plt.axes([0.1, 0, 0.1, 0.075])
    b_rand = Button(ax, 'Rand')
    b_rand.on_clicked(random_memory)
    ax = plt.axes([0.0, 0, 0.1, 0.075])
    b_inv = Button(ax, 'Inv')
    b_inv.on_clicked(invert_network)
    ax = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
    s_temp = Slider(ax, label="T", valmin=0, valmax=50, valinit=0, orientation="vertical",valfmt="%i")
    s_temp.on_changed(change_temp)



    s_rect = SelectionRect(imgax, fig.canvas, on_box_release)
    s_rect.connect()

    plt.show()



def SpinGlass5by5():
    N = 5
    h = Hopfield(N)
    s_circle = np.array([[1,1,1,1,1],
    [1,-1,-1,-1,1],
    [1,-1,-1,-1,1],
    [1,-1,-1,-1,1],
    [1,1,1,1,1]])
    s_cross = np.array([[1,-1,-1,-1,1],
    [-1,1,-1,1,-1],
    [-1,-1,1,-1,-1],
    [-1,1,-1,1,-1],
    [1,-1,-1,-1,1]])
    s_phi = np.array([[1,-1,1,1,1],
    [1,-1,1,-1,1],
    [1,1,1,1,1],
    [-1,-1,1,-1,-1],
    [-1,-1,1,-1,-1]])
    s_fence = np.array([[1,-1,1,-1,1],
    [1,-1,1,-1,1],
    [1,-1,1,-1,1],
    [1,-1,1,-1,1],
    [1,-1,1,-1,1]])
    s_S = np.array([[1,1,1,1,1],
    [1,-1,-1,-1,-1],
    [1,1,1,1,1],
    [-1,-1,-1,-1,1],
    [1,1,1,1,1]])
    s_dot = np.array([[-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1]])

    h.add_shape(s_phi)
    h.add_shape(s_cross)
    h.train_all_shapes()
    h.set_network(h.shapes[0])

    fig, imgax = plt.subplots()
    imgax.set_position([0.1,0.16,0.8,0.8])
    pltimage = imgax.imshow(h.network, cmap=plt.get_cmap("binary"))
    
    def plt_do_sync_iteration(event):
        h.sync_update()
        pltimage.set_data(h.network)
        plt.draw()

    def plt_train_2_shapes(event):
        h.reset_shapes()
        h.add_shape(s_phi)
        h.add_shape(s_cross)
        h.train_all_shapes()
        plt.title("2 shapes\n(ɑ=0.08)")

    def plt_train_6_shapes(event):
        h.reset_shapes()
        h.add_shape(s_circle)
        h.add_shape(s_cross)
        h.add_shape(s_phi)
        h.add_shape(s_fence)
        h.add_shape(s_S)
        h.add_shape(s_dot)
        plt.title("6 shapes\n(ɑ=0.24)")
        h.train_all_shapes()

    def plt_do_ordered_async_iteration(event):
        h.async_update_ordered(pause_and_update)
        pltimage.set_data(h.network)
        plt.draw()
        
    def random_memeory(event):
        h.network = h.shapes[randint(0,len(h.shapes)-1)]
        pltimage.set_data(h.network)
        plt.draw()
    
    def invert_network(event):
        h.network = h.network*-1
        pltimage.set_data(h.network)
        plt.draw()
    
    def pause_and_update():
        plt.pause(0.01)
        pltimage.set_data(h.network)
        plt.draw()

    def on_box_release(pos, size):
        image_box_cleared = h.network.copy()
        size = [min(x, h.N) for x in size]
        if size[0] == 1 and size[1] == 1:
            image_box_cleared[pos[1]][pos[0]] = 1 if image_box_cleared[pos[1]][pos[0]] == -1 else -1
        else:
            for x in range(pos[0],pos[0]+size[0]):
                for y in range(pos[1],pos[1]+size[1]):
                    image_box_cleared[y][x] = 1
        h.set_network(image_box_cleared)
        pltimage.set_data(h.network)
        plt.draw()
        
    ax = plt.axes([0.0, 0, 0.1, 0.075])
    b_inv = Button(ax, 'Inv')
    b_inv.on_clicked(invert_network)
    ax = plt.axes([0.1, 0, 0.1, 0.075])
    b_rand = Button(ax, 'Rand')
    b_rand.on_clicked(random_memeory)
    ax = plt.axes([0.2, 0, 0.2, 0.075])
    b_2shapes = Button(ax, '2 Shapes')
    b_2shapes.on_clicked(plt_train_2_shapes)
    ax = plt.axes([0.4, 0, 0.2, 0.075])
    b_6shapes = Button(ax, '6 Shapes')
    b_6shapes.on_clicked(plt_train_6_shapes)
    ax = plt.axes([0.6, 0, 0.2, 0.075])
    bsync = Button(ax, 'Do Sync')
    bsync.on_clicked(plt_do_sync_iteration)
    ax = plt.axes([0.8, 0, 0.2, 0.075])
    b_order = Button(ax, 'Do Ordered')
    b_order.on_clicked(plt_do_ordered_async_iteration)

    s_rect = SelectionRect(imgax, fig.canvas, on_box_release)
    s_rect.connect()

    plt.title("2 shapes\n(ɑ=0.08)")
    plt.show()



SpinGlass5by5()
# asterixObelix()
