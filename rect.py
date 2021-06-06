from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class SelectionRect:
    def __init__(self, ax, canvas, box_callback):
        self.ax = ax
        self.canvas = canvas
        self.box_callback = box_callback
        self.press = None
        self.dragpos = [[0,0],[0,0]]
        
    def on_press(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
        if event.inaxes != self.ax:
            return
        self.press = 1
        self.dragpos = [[round(event.xdata), round(event.ydata)],[round(event.xdata), round(event.ydata)]]
        # self.rectStart = [event.xdata, event.ydata]
        self.rect = Rectangle((self.dragpos[0][0]-0.5, self.dragpos[0][1]-0.5),0,0,0, alpha=0.3)
        self.rect.set_alpha(0.4)
        self.ax.add_patch(self.rect)
        
        # self.ax.contourf([[0,0],[event.xdata, event.ydata]])
    def on_release(self, event):
        if self.press == None:
            return
        self.rect.remove()
        self.press = None
        x = self.rect.xy[0]+0.5
        y = self.rect.xy[1]+0.5
        width = self.rect.get_width()
        height = self.rect.get_height()
        self.box_callback([round(x),round(y)], [round(width+1),round(height+1)])
        
        print("BOX: x:{},y:{}   w:{}h:{}".format(self.rectStart[0], self.rectStart[0], width, height))
    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            print("{}{}".format(self.ax, event.inaxes))
            return
        self.dragpos[1] = [round(event.xdata), round(event.ydata)]
        dp = self.dragpos
        xmin = min(dp[0][0], dp[1][0])
        ymin = min(dp[0][1], dp[1][1])
        xmax = max(dp[0][0], dp[1][0])
        ymax = max(dp[0][1], dp[1][1])
        self.rect.set_width( round(xmax-xmin))
        self.rect.set_height( round(ymax-ymin))
        self.rect.xy = (xmin-0.5, ymin-0.5)
        plt.draw()
        # print("motion {},{}".format(event.xdata,event.ydata))
    def connect(self):
        self.cidpress = self.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)