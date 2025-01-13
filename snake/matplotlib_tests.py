import matplotlib.pyplot as plt
from IPython import display
import time
import random


# Enable interactive mode
plt.ion()

def plot(fig, spec, x, y):
  
  plt.clf()

  fig.suptitle('Simulation Metrics')
  ax0 = fig.add_subplot(spec[0])
  annotate_axes(ax0, 'Game Times')
  ax1 = fig.add_subplot(spec[1])
  annotate_axes(ax1, 'Game Scores')

  ax0.plot(x, 'b-')
  ax1.plot(y, 'b-')
  
  plt.show()
  plt.pause(0.1)

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")
    
def main_loop():
  xdata = []
  ydata = []
  # Layouts: constrained, compressed, tight
  fig = plt.figure(figsize=(5,3), layout="compressed")
  
  spec = fig.add_gridspec(ncols=1, nrows=2)
  
  for x in range(10):
    xdata.append(random.randint(0,x))
    ydata.append(random.randint(0,x))
    plot(fig, spec, xdata, ydata)
    time.sleep(1)

def main_loop2():
  xdata = []
  ydata = []
  fig, axes = plt.subplots(ncols=1,nrows=2, 
                           figsize=(5.5, 3.5), 
                           layout="compressed")
  
  for x in range(10):
    xdata.append(random.randint(0,x))
    ydata.append(random.randint(0,x))
    plot(fig, axes, xdata, ydata)
    time.sleep(1)
  

if __name__ == '__main__':
  main_loop()