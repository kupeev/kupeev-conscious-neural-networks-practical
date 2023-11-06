import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.widgets import Button
import builtins

is_notebook = hasattr(builtins, "__IPYTHON__")
if is_notebook:
    if 1: #with this displays as sep wnd in Jup
        # https://stackoverflow.com/questions/37365357/when-i-use-matplotlib-in-jupyter-notebook-it-always-raise-matplotlib-is-curren
        import matplotlib
        matplotlib.use('TkAgg')
    elif 0:
        #does not work in Jup
        #  UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
        pass
else:
    pass

#https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
#   Following:
#   https://matplotlib.org/stable/gallery/widgets/buttons.html
height = width =20


fig = plt.figure()
ax1 = fig.add_subplot(121)#
ax2 = fig.add_subplot(122)
plt.ion() # Turns interactive mode on (probably unnecessary)
fig.show() # Initially shows the figure

button_stop_clicked = 0
button_pauseresume_clicked = 0

class Index:
    ind = 0

    def pauseresume(self, event):
        global button_pauseresume_clicked
        button_pauseresume_clicked = 1
        #plt.draw()

    def stop(self, event):
        global button_stop_clicked
        button_stop_clicked = 1
        #plt.draw()

callback = Index()

ax_pauseresume = plt.axes([0.55, 0.05, 0.2, 0.075])
ax_stop = plt.axes([0.81, 0.05, 0.1, 0.075])

btn_pauseresume = Button(ax_pauseresume, 'Pause/Resume')
btn_pauseresume.on_clicked(callback.pauseresume)

btn_stop = Button(ax_stop, 'Stop')
btn_stop.on_clicked(callback.stop)

paused = 0
while(1):
    if paused:
        #print('paused')
        plt.pause(.1) # Delay in seconds
        fig.canvas.draw() # Draws the image to the screen
    else:
        ax1.clear() # Clears the previous image
        ax2.clear() # Clears the previous image
        frame1 = np.random.randint(7, size=(height,width))
        ax1.imshow(frame1) # Loads the new image
        frame2 = np.random.randint(7, size=(height,width))
        ax2.imshow(frame2) # Loads the new image
        plt.pause(.1) # Delay in seconds
        fig.canvas.draw() # Draws the image to the screen

    if button_stop_clicked:
        break
    if button_pauseresume_clicked:
        print('button_pauseresume_clicked='+str(button_pauseresume_clicked)+' paused='+str(paused))
        if 1:
            if paused:
                paused = 0
            else:
                paused = 1
        button_pauseresume_clicked = 0
#while(1):

plt.close('all')

