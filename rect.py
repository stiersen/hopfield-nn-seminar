class SelectionRect:
    def __init__(self, ax, canvas, box_callback):
        self.ax = ax
        self.canvas = canvas
        self.box_callback = box_callback
        self.press = None
        self.rectStart = []
        
    def on_press(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
        if event.inaxes != self.ax:
            return
        self.press = 1
        self.rectStart = [event.xdata, event.ydata]
    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        self.press = None
        x = min(self.rectStart[0], event.xdata)
        y = min(self.rectStart[1], event.ydata)
        width = abs(event.xdata - self.rectStart[0])
        height = abs(event.ydata - self.rectStart[1])
        self.box_callback([round(x),round(y)], [round(width),round(height)])
        
        print("BOX: x:{},y:{}   w:{}h:{}".format(self.rectStart[0], self.rectStart[0], width, height))
    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            print("{}{}".format(self.ax, event.inaxes))
            return
        print("motion {},{}".format(event.xdata,event.xdata))
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